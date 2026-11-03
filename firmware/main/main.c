// main.c — ESP32-S3 side: memory placement, dual-core dispatch, serial protocol.
//
// Protocol (lines over UART0 or USB-Serial-JTAG, whichever the host talks to):
//   INFO                                   -> @INFO key=value ...
//   GEN <mode> <steps> <verify> <id,id,..> -> @T <argmax> <us> [<top1> <margin>] ... @END ...
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_log.h"
#include "esp_chip_info.h"
#include "esp_rom_sys.h"
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"
#include "llm.h"

static const char *TAG = "llm";

// ───────────── memory ─────────────
static void *alloc_int(size_t n) { return heap_caps_aligned_calloc(16, 1, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); }
static void *alloc_psram(size_t n) { return heap_caps_aligned_calloc(16, 1, n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }
static uint32_t cycles(void) { return esp_cpu_get_cycle_count(); }

// ───────────── second core: spin-waiting worker ─────────────
static volatile llm_job_fn job_fn;
static volatile int job_a, job_b, worker_spin;
static volatile uint32_t job_req, job_done;
static TaskHandle_t worker;

static void worker_task(void *arg) {
    uint32_t seen = job_req;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);       // sleep while nobody needs core 1
        while (worker_spin) {
            if (job_req != seen) {
                seen = job_req;
                job_fn(job_a, job_b);
                __sync_synchronize();
                job_done = seen;
            }
        }
    }
}

static void IRAM_ATTR run_parallel(llm_job_fn fn, int n) {
    int h = n / 2;
    job_fn = fn; job_a = h; job_b = n;
    __sync_synchronize();
    job_req = job_req + 1;
    fn(0, h);
    while (job_done != job_req) { }
    __sync_synchronize();
}

// ───────────── serial I/O on both ports ─────────────
static int usj_ok;
static int out_port;              // 0 = UART0, 1 = USB-Serial-JTAG

static void out(const char *fmt, ...) {
    char buf[256];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > (int)sizeof buf - 1) n = sizeof buf - 1;
    if (out_port == 1 && usj_ok) usb_serial_jtag_write_bytes(buf, n, pdMS_TO_TICKS(200));
    else uart_write_bytes(UART_NUM_0, buf, n);
}

// ───────────── models & modes ─────────────
typedef struct { int16_t tok; int32_t us; float top, mg; } step_t;
static step_t *steps_buf;
static llm_model_t m_flash, m_sram, m_psram, m_f32;
static int have_sram, have_psram, have_f32;
static llm_state_t st;
static const uint8_t *blob;

typedef struct { const llm_model_t *m; llm_kern_t kern; int cores; int *avail; int attn; } mode_t_;
static int always = 1;
#define NMODES 8
static mode_t_ MODES[NMODES] = {
    { &m_f32,   KERN_F32,    1, &have_f32 },
    { &m_psram, KERN_SCALAR, 1, &have_psram },
    { &m_flash, KERN_SCALAR, 1, &always },
    { &m_sram,  KERN_SCALAR, 1, &have_sram },
    { &m_sram,  KERN_PIE,    1, &have_sram },
    { &m_sram,  KERN_PIE,    2, &have_sram },
    { &m_sram,  KERN_PIE,    2, &have_sram, 1 },   // + attention v2 (GQA pairs, registers)
    { &m_sram,  KERN_PIE,    2, &have_sram, 2 },   // + integer Q.K
};

static int ids[512];

static void cmd_info(void) {
    char modes[32] = "";
    for (int i = 0; i < NMODES; i++)
        if (*MODES[i].avail) { char t[4]; snprintf(t, 4, "%s%d", modes[0] ? "," : "", i); strcat(modes, t); }
    const llm_cfg_t *c = &m_flash.c;
    out("@INFO chip=esp32s3 cpu_mhz=%lu dim=%d layers=%d vocab=%d seq=%d blob_kb=%u modes=%s "
        "free_int_kb=%u free_psram_kb=%u psram_total_kb=%u\n",
        (unsigned long)esp_rom_get_cpu_ticks_per_us(), c->dim, c->L, c->vocab, st.kv_seq, (unsigned)(m_flash.blob_size / 1024), modes,
        (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
        (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
        (unsigned)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024));
}

static void cmd_gen(char *args) {
    int mode = 0, steps = 0, verify = 0, n = 0;
    char *p = args;
    mode = strtol(p, &p, 10); steps = strtol(p, &p, 10); verify = strtol(p, &p, 10);
    while (*p == ' ') p++;
    while (*p && n < 512) { ids[n++] = strtol(p, &p, 10); if (*p == ',') p++; else break; }
    if (mode < 0 || mode >= NMODES || !*MODES[mode].avail) { out("!ERR mode %d not available\n", mode); return; }
    if (n < 1) { out("!ERR no prompt\n"); return; }
    if (steps > st.kv_seq) steps = st.kv_seq;

    st.m = MODES[mode].m; st.kern = MODES[mode].kern; st.ncores = MODES[mode].cores; st.attn = MODES[mode].attn;
    st.cyc_mm = st.cyc_att = st.cyc_all = 0;
    llm_reset(&st);
    if (st.ncores > 1) { worker_spin = 1; xTaskNotifyGive(worker); vTaskDelay(1); }

    int tok = ids[0], done = 0;
    int64_t t_all = esp_timer_get_time();
    for (int pos = 0; pos < steps; pos++) {
        float top, mg;
        int64_t t0 = esp_timer_get_time();
        int nx = llm_forward(&st, tok, pos, &top, &mg);
        int32_t us = (int32_t)(esp_timer_get_time() - t0);
        steps_buf[pos] = (step_t){ (int16_t)nx, us, top, mg };
        done = pos + 1;
        tok = pos + 1 < n ? ids[pos + 1] : nx;
        if (tok == 1) break;
    }
    t_all = esp_timer_get_time() - t_all;
    worker_spin = 0;

    for (int i = 0; i < done; i++) {
        step_t *s = &steps_buf[i];
        if (verify) out("@T %d %ld %.9g %.9g\n", s->tok, (long)s->us, (double)s->top, (double)s->mg);
        else out("@T %d %ld\n", s->tok, (long)s->us);
    }
    uint32_t all = st.cyc_all ? st.cyc_all : 1;
    unsigned mm = (unsigned)(100ull * st.cyc_mm / all), att = (unsigned)(100ull * st.cyc_att / all);
    out("@END n=%d us=%lld mm=%u att=%u oth=%u mode=%d\n", done, (long long)t_all, mm, att,
        100 - mm - att > 100 ? 0 : 100 - mm - att, mode);
}

static void handle(char *line, int port) {
    out_port = port;
    if (!strncmp(line, "INFO", 4)) cmd_info();
    else if (!strncmp(line, "GEN ", 4)) cmd_gen(line + 4);
    else if (line[0]) out("!ERR unknown command\n");
}

void app_main(void) {
    // core-1 worker first, before the big allocations eat the heap
    xTaskCreatePinnedToCore(worker_task, "core1", 3072, NULL, configMAX_PRIORITIES - 2, &worker, 1);

    // serial drivers (small buffers: every KB of SRAM matters here)
    uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0);
    usb_serial_jtag_driver_config_t ucfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ucfg.rx_buffer_size = 512; ucfg.tx_buffer_size = 1024;
    usj_ok = usb_serial_jtag_driver_install(&ucfg) == ESP_OK;

    ESP_LOGI(TAG, "free internal %u KB (largest %u), psram %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    // model blob lives in its own flash partition and is memory-mapped (XIP)
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    esp_partition_mmap_handle_t h;
    const void *map = NULL;
    int ok = part && esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &map, &h) == ESP_OK;
    if (!ok || llm_parse(&m_flash, map, part->size) != 0) {
        for (;;) {
            out("!NOMODEL flash model.espl to the 'model' partition (python esp_llm.py flash)\n");
            vTaskDelay(pdMS_TO_TICKS(2000));
        }
    }
    blob = map;
    llm_alloc = alloc_int;
    llm_cycles = cycles;
    llm_parallel = run_parallel;

    // SRAM budget: weights + activations + int8 KV cache must fit together.
    // Shrink the KV cache (max context) until they do.
    const llm_cfg_t *c = &m_flash.c;
    size_t wbytes = llm_weight_bytes(&m_flash), margin = 12 * 1024;
    size_t free_int = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    int kv_seq = c->seq;
    while (kv_seq > 32 && wbytes + llm_state_bytes(c, kv_seq) + c->seq * sizeof(step_t) + margin > free_int)
        kv_seq -= 32;
    int fits = wbytes + llm_state_bytes(c, kv_seq) + c->seq * sizeof(step_t) + margin <= free_int;
    if (!fits) kv_seq = c->seq;          // no SRAM mode; give the KV cache the full context
    if (llm_state_init(&st, &m_flash, kv_seq) != 0) ESP_LOGE(TAG, "state alloc failed");
    st.m = &m_flash;
    steps_buf = heap_caps_calloc(kv_seq, sizeof(step_t), MALLOC_CAP_INTERNAL);

    if (fits) have_sram = llm_relocate(&m_sram, &m_flash, alloc_int, 0) == 0;
    if (heap_caps_get_total_size(MALLOC_CAP_SPIRAM) > 0) {
        have_psram = llm_relocate(&m_psram, &m_flash, alloc_psram, 0) == 0;
        have_f32 = llm_relocate(&m_f32, &m_flash, alloc_psram, 1) == 0;
    }
    ESP_LOGI(TAG, "weights %u KB, kv_seq %d, placement: sram=%d psram=%d fp32=%d, free internal %u KB",
             (unsigned)(wbytes / 1024), kv_seq, have_sram, have_psram, have_f32,
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));

    vTaskPrioritySet(NULL, configMAX_PRIORITIES - 2);
    esp_log_level_set("*", ESP_LOG_WARN);
    out_port = 0; cmd_info();
    if (usj_ok) { out_port = 1; cmd_info(); }

    static char lb[2][2048];  // prompt of up to ~400 token ids
    static int ln[2];
    uint8_t buf[256];
    for (;;) {
        for (int port = 0; port < 2; port++) {
            int r = port == 0 ? uart_read_bytes(UART_NUM_0, buf, sizeof buf, pdMS_TO_TICKS(5))
                              : (usj_ok ? usb_serial_jtag_read_bytes(buf, sizeof buf, pdMS_TO_TICKS(5)) : 0);
            for (int i = 0; i < r; i++) {
                char ch = buf[i];
                if (ch == '\n' || ch == '\r') {
                    lb[port][ln[port]] = 0;
                    if (ln[port]) handle(lb[port], port);
                    ln[port] = 0;
                } else if (ln[port] < (int)sizeof lb[0] - 1) lb[port][ln[port]++] = ch;
            }
        }
    }
}
