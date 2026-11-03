// llm.h — tiny int8 Llama engine for ESP32-S3 (portable core, builds on host too)
#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct {
    int dim, hidden, hp, L, nh, nkv, vocab, seq, kv, hs;
} llm_cfg_t;

typedef struct {
    const int8_t *q;     // rows x cols, row-major, 16-byte aligned rows
    const float  *s;     // per-row scale
    const float  *f;     // optional dequantized fp32 copy (mode 0)
    int rows, cols;
} qmat_t;

typedef struct {
    const float *rms_att, *rms_ffn;
    qmat_t qkv, wo, w13, w2;
} llm_layer_t;

#define LLM_MAX_LAYERS 32

typedef struct {
    llm_cfg_t c;
    qmat_t emb;                 // tied: token embedding == classifier
    const float *rms_final;
    const float *rope;          // [seq][hs/2][2] cos,sin
    llm_layer_t layer[LLM_MAX_LAYERS];
    size_t blob_size;
} llm_model_t;

typedef enum { KERN_F32 = 0, KERN_SCALAR = 1, KERN_PIE = 2 } llm_kern_t;

typedef struct {
    const llm_model_t *m;
    llm_kern_t kern;
    int ncores;
    int attn;                    // 0 = per-head v1, 1 = GQA-pair v2 (bit-identical), 2 = v2 + int8 Q.K
    int kv_seq;                  // KV cache length (may be < model seq to fit SRAM)
    // activations
    float *x, *xb, *qkv, *h13, *hh, *att_out, *logits;
    float *att[2];               // per-core softmax scratch
    int8_t *xq;                  // quantized activation (16-aligned)
    int32_t *acc;                // int32 accumulators
    // int8 KV cache
    int8_t *k8, *v8;
    float *ks, *vs;
    // profiling (cycles)
    uint32_t cyc_mm, cyc_att, cyc_all;
} llm_state_t;

// platform hooks (set by main.c on the device, left NULL on the host)
typedef void (*llm_job_fn)(int a, int b);
extern void (*llm_parallel)(llm_job_fn fn, int n);   // runs fn(0,n/2) here and fn(n/2,n) on core 1
extern uint32_t (*llm_cycles)(void);
extern void *(*llm_alloc)(size_t size);              // 16-byte aligned scratch allocator

int  llm_parse(llm_model_t *m, const uint8_t *blob, size_t len);
// copy every tensor into memory returned by alloc(); with to_f32 also builds fp32 matrices
int  llm_relocate(llm_model_t *dst, const llm_model_t *src, void *(*alloc)(size_t), int to_f32);
int  llm_state_init(llm_state_t *s, const llm_model_t *m, int kv_seq);
size_t llm_state_bytes(const llm_cfg_t *c, int kv_seq);
size_t llm_weight_bytes(const llm_model_t *m);   // what llm_relocate(.., to_f32=0) allocates
void llm_reset(llm_state_t *s);
// one forward step; returns argmax token, writes top logit and margin to top1 / margin
int  llm_forward(llm_state_t *s, int tok, int pos, float *top1, float *margin);
