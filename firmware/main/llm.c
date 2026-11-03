// llm.c — int8 Llama-2 forward pass tuned for ESP32-S3.
// Numerics are mirrored op-by-op by Q8Sim in esp_llm.py, so host and device can be compared exactly.
#include "llm.h"
#include <string.h>
#include <math.h>

#ifdef LLM_HAVE_PIE
// mv_pie.S: acc[r] = sum_j W[r][j] * x[j]   (int8 x int8 -> int32), cols multiple of 16
void mv_s8_pie(const int8_t *W, const int8_t *x, int32_t *acc, int rows, int n16);
void mv_s8_pie_64(const int8_t *W, const int8_t *x, int32_t *acc, int rows);
#endif

#ifndef HOT
#define HOT
#endif

void (*llm_parallel)(llm_job_fn fn, int n) = 0;
uint32_t (*llm_cycles)(void) = 0;
void *(*llm_alloc)(size_t size) = 0;

static inline uint32_t cyc(void) { return llm_cycles ? llm_cycles() : 0; }

// ───────────────────────── blob parsing ─────────────────────────

typedef struct { const uint8_t *p; size_t off, len; } rd_t;

static const void *take(rd_t *r, size_t bytes) {
    if (r->off + bytes > r->len) return 0;
    const void *p = r->p + r->off;
    r->off += (bytes + 15) & ~(size_t)15;
    return p;
}

static int take_mat(rd_t *r, qmat_t *w, int rows, int cols) {
    w->rows = rows; w->cols = cols; w->f = 0;
    w->q = take(r, (size_t)rows * cols);
    w->s = take(r, (size_t)rows * 4);
    return w->q && w->s;
}

int llm_parse(llm_model_t *m, const uint8_t *blob, size_t len) {
    const uint32_t *h = (const uint32_t *)blob;
    if (len < 64 || memcmp(blob, "ESPL", 4) != 0 || h[1] != 1) return -1;
    llm_cfg_t *c = &m->c;
    c->dim = h[2]; c->hidden = h[3]; c->hp = h[4]; c->L = h[5]; c->nh = h[6]; c->nkv = h[7];
    c->vocab = h[8]; c->seq = h[9]; c->kv = h[10]; c->hs = h[11];
    m->blob_size = h[14];
    if (c->L > LLM_MAX_LAYERS || m->blob_size > len) return -2;
    rd_t r = { blob, 64, m->blob_size };
    int D = c->dim, kv = c->kv, Hp = c->hp;
    int ok = take_mat(&r, &m->emb, c->vocab, D);
    ok &= (m->rms_final = take(&r, D * 4)) != 0;
    for (int l = 0; l < c->L; l++) {
        llm_layer_t *ly = &m->layer[l];
        ok &= (ly->rms_att = take(&r, D * 4)) != 0;
        ok &= (ly->rms_ffn = take(&r, D * 4)) != 0;
        ok &= take_mat(&r, &ly->qkv, D + 2 * kv, D);
        ok &= take_mat(&r, &ly->wo, D, D);
        ok &= take_mat(&r, &ly->w13, 2 * Hp, D);
        ok &= take_mat(&r, &ly->w2, D, Hp);
    }
    ok &= (m->rope = take(&r, (size_t)c->seq * c->hs * 4)) != 0;
    return ok ? 0 : -3;
}

static const void *dup(const void *src, size_t n, void *(*alloc)(size_t)) {
    void *p = alloc(n);
    if (p) memcpy(p, src, n);
    return p;
}

static int reloc_mat(qmat_t *d, const qmat_t *s, void *(*alloc)(size_t), int to_f32) {
    *d = *s;
    size_t n = (size_t)s->rows * s->cols;
    d->s = dup(s->s, s->rows * 4, alloc);
    if (to_f32) {
        float *f = alloc(n * 4);
        if (!f) return 0;
        for (int r = 0; r < s->rows; r++)
            for (int j = 0; j < s->cols; j++)
                f[(size_t)r * s->cols + j] = (float)s->q[(size_t)r * s->cols + j] * s->s[r];
        d->f = f;
        d->q = s->q;            // int8 copy not needed for fp32 mode (embedding lookup still uses it)
        return d->s != 0;
    }
    d->q = dup(s->q, n, alloc);
    return d->q && d->s;
}

int llm_relocate(llm_model_t *d, const llm_model_t *s, void *(*alloc)(size_t), int to_f32) {
    *d = *s;
    const llm_cfg_t *c = &s->c;
    int D = c->dim, ok = reloc_mat(&d->emb, &s->emb, alloc, to_f32);
    ok &= (d->rms_final = dup(s->rms_final, D * 4, alloc)) != 0;
    for (int l = 0; l < c->L && ok; l++) {
        llm_layer_t *dl = &d->layer[l]; const llm_layer_t *sl = &s->layer[l];
        ok &= (dl->rms_att = dup(sl->rms_att, D * 4, alloc)) != 0;
        ok &= (dl->rms_ffn = dup(sl->rms_ffn, D * 4, alloc)) != 0;
        ok &= reloc_mat(&dl->qkv, &sl->qkv, alloc, to_f32);
        ok &= reloc_mat(&dl->wo, &sl->wo, alloc, to_f32);
        ok &= reloc_mat(&dl->w13, &sl->w13, alloc, to_f32);
        ok &= reloc_mat(&dl->w2, &sl->w2, alloc, to_f32);
    }
    d->rope = s->rope;   // stays in flash: one 32-byte row per token, the cache takes care of it
    return ok ? 0 : -1;
}

// ───────────────────────── state ─────────────────────────

static size_t al16(size_t n) { return (n + 15) & ~(size_t)15; }

size_t llm_state_bytes(const llm_cfg_t *c, int kv_seq) {
    int D = c->dim, maxv = D > c->hp ? D : c->hp;
    int maxr = c->vocab;
    if (2 * c->hp > maxr) maxr = 2 * c->hp;
    if (D + 2 * c->kv > maxr) maxr = D + 2 * c->kv;
    size_t kvn = (size_t)c->L * kv_seq * c->kv;
    return 3 * al16(D * 4) + al16((D + 2 * c->kv) * 4) + al16(2 * c->hp * 4) + al16(c->hp * 4) + al16(c->vocab * 4)
         + 2 * al16(2 * kv_seq * 4) + al16(maxv) + al16(maxr * 4) + 2 * al16(kvn) + 2 * al16((size_t)c->L * kv_seq * 4);
}

size_t llm_weight_bytes(const llm_model_t *m) {
    const llm_cfg_t *c = &m->c;
    size_t D = c->dim, kv = c->kv, Hp = c->hp, n = al16(c->vocab * D) + al16(c->vocab * 4) + al16(D * 4);
    n += c->L * (2 * al16(D * 4) + al16((D + 2 * kv) * D) + al16((D + 2 * kv) * 4) + al16(D * D) + al16(D * 4)
                 + al16(2 * Hp * D) + al16(2 * Hp * 4) + al16(D * Hp) + al16(D * 4));
    return n;
}

int llm_state_init(llm_state_t *s, const llm_model_t *m, int kv_seq) {
    const llm_cfg_t *c = &m->c;
    memset(s, 0, sizeof(*s));
    s->m = m;
    if (kv_seq <= 0 || kv_seq > c->seq) kv_seq = c->seq;
    s->kv_seq = kv_seq;
    int D = c->dim, maxv = D > c->hp ? D : c->hp;
    int maxr = c->vocab;
    if (2 * c->hp > maxr) maxr = 2 * c->hp;
    if (D + 2 * c->kv > maxr) maxr = D + 2 * c->kv;
    size_t kvn = (size_t)c->L * kv_seq * c->kv;
#define A(ptr, n) if (!((ptr) = llm_alloc(n))) return -1
    A(s->x, D * 4); A(s->xb, D * 4); A(s->att_out, D * 4);
    A(s->qkv, (D + 2 * c->kv) * 4); A(s->h13, 2 * c->hp * 4); A(s->hh, c->hp * 4);
    A(s->logits, c->vocab * 4);
    A(s->att[0], 2 * kv_seq * 4); A(s->att[1], 2 * kv_seq * 4);
    A(s->xq, (maxv + 15) & ~15); A(s->acc, maxr * 4);
    A(s->k8, kvn); A(s->v8, kvn);
    A(s->ks, (size_t)c->L * kv_seq * 4); A(s->vs, (size_t)c->L * kv_seq * 4);
#undef A
    s->kern = KERN_SCALAR; s->ncores = 1;
    return 0;
}

void llm_reset(llm_state_t *s) {
    const llm_cfg_t *c = &s->m->c;
    size_t kvn = (size_t)c->L * s->kv_seq * c->kv;
    memset(s->k8, 0, kvn); memset(s->v8, 0, kvn);
}

// ───────────────────────── math helpers ─────────────────────────

#define EC0 1.0000035762786865f
#define EC1 0.692969560623169f
#define EC2 0.24162131547927856f
#define EC3 0.05171773582696915f
#define EC4 0.013683983124792576f

// exp(x) = 2^(x*log2e): split into integer part (exponent bits) and fraction (deg-4 polynomial)
static inline float fast_expf(float x) {
    if (x < -87.0f) x = -87.0f;
    if (x > 88.0f) x = 88.0f;
    float t = x * 1.442695041f;
    int i = (int)t;
    if (t < (float)i) i--;
    float f = t - (float)i;
    float p = EC0 + f * (EC1 + f * (EC2 + f * (EC3 + f * EC4)));
    union { int32_t i; float f; } u;
    u.i = (int32_t)((uint32_t)(i + 127) << 23);
    return p * u.f;
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int j = 0; j < n; j++) ss += x[j] * x[j];
    ss = ss / (float)n + 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < n; j++) o[j] = w[j] * (ss * x[j]);
}

static inline int round_away(float v) { return (int)(v + (v >= 0.0f ? 0.5f : -0.5f)); }

// dynamic symmetric int8 quantization of an activation vector; returns scale
static float q8_vec(int8_t *q, const float *x, int n) {
    float amax = 0.0f;
    for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > amax) amax = a; }
    if (amax == 0.0f) { memset(q, 0, n); return 1.0f; }
    float inv = 127.0f / amax, s = amax / 127.0f;
    for (int j = 0; j < n; j++) {
        int v = round_away(x[j] * inv);
        q[j] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
    return s;
}

static void rope(float *v, int nheads, int hs, const float *cs) {
    for (int h = 0; h < nheads; h++)
        for (int i = 0; i < hs / 2; i++) {
            float *p = v + h * hs + 2 * i;
            float c = cs[2 * i], s = cs[2 * i + 1], a = p[0], b = p[1];
            p[0] = a * c - b * s;
            p[1] = a * s + b * c;
        }
}

// ───────────────────────── matmul (row ranges, so two cores can split it) ─────────────────────────

static struct {
    llm_state_t *s; const qmat_t *W; const float *xf; float *out; float sx;
    int l, pos;
} G;

static HOT void mm_rows(int r0, int r1) {
    const qmat_t *W = G.W;
    int n = W->cols;
    llm_state_t *s = G.s;
    if (s->kern == KERN_F32) {
        for (int r = r0; r < r1; r++) {
            const float *w = W->f + (size_t)r * n;
            float acc = 0.0f;
            for (int j = 0; j < n; j++) acc += w[j] * G.xf[j];
            G.out[r] = acc;
        }
        return;
    }
    int32_t *acc = s->acc;
#ifdef LLM_HAVE_PIE
    if (s->kern == KERN_PIE) {
        if (n == 64) mv_s8_pie_64(W->q + (size_t)r0 * n, s->xq, acc + r0, r1 - r0);
        else mv_s8_pie(W->q + (size_t)r0 * n, s->xq, acc + r0, r1 - r0, n >> 4);
    } else
#endif
    {
        const int8_t *x = s->xq;
        for (int r = r0; r < r1; r++) {
            const int8_t *w = W->q + (size_t)r * n;
            int32_t a = 0;
            for (int j = 0; j < n; j += 4)
                a += w[j] * x[j] + w[j + 1] * x[j + 1] + w[j + 2] * x[j + 2] + w[j + 3] * x[j + 3];
            acc[r] = a;
        }
    }
    for (int r = r0; r < r1; r++) G.out[r] = ((float)acc[r] * W->s[r]) * G.sx;
}

static void matmul(llm_state_t *s, float *out, const qmat_t *W, const float *x) {
    uint32_t t0 = cyc();
    G.s = s; G.W = W; G.out = out; G.xf = x;
    if (s->kern != KERN_F32) G.sx = q8_vec(s->xq, x, W->cols);
    if (s->ncores > 1 && llm_parallel) llm_parallel(mm_rows, W->rows);
    else mm_rows(0, W->rows);
    s->cyc_mm += cyc() - t0;
}

// ───────────────────────── attention over the int8 KV cache ─────────────────────────

static HOT void attn_heads(int h0, int h1) {
    llm_state_t *s = G.s;
    const llm_cfg_t *c = &s->m->c;
    int hs = c->hs, kvd = c->kv, rep = c->nh / c->nkv, pos = G.pos, l = G.l;
    float *att = s->att[h0 == 0 ? 0 : 1];
    const int8_t *K = s->k8 + (size_t)l * s->kv_seq * kvd;
    const int8_t *V = s->v8 + (size_t)l * s->kv_seq * kvd;
    const float *ks = s->ks + (size_t)l * s->kv_seq, *vs = s->vs + (size_t)l * s->kv_seq;
    float isq = 1.0f / sqrtf((float)hs);
    float qh[64];
    for (int h = h0; h < h1; h++) {
        int kh = h / rep;
        for (int d = 0; d < hs; d++) qh[d] = s->qkv[h * hs + d] * isq;
        float mx = -1e30f;
        for (int t = 0; t <= pos; t++) {
            const int8_t *k = K + (size_t)t * kvd + kh * hs;
            float a = 0.0f;
            for (int d = 0; d < hs; d++) a += (float)k[d] * qh[d];
            a = a * ks[t];
            att[t] = a;
            if (a > mx) mx = a;
        }
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) { att[t] = fast_expf(att[t] - mx); sum += att[t]; }
        float inv = 1.0f / sum;
        float *o = s->att_out + h * hs;
        for (int d = 0; d < hs; d++) o[d] = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float w = (att[t] * inv) * vs[t];
            const int8_t *v = V + (size_t)t * kvd + kh * hs;
            for (int d = 0; d < hs; d++) o[d] += w * (float)v[d];
        }
    }
}


// ── attention v2: one pass over the KV cache serves both query heads of a GQA group ──
// Each int8 K/V element is loaded and converted to float once instead of twice, the 8-wide
// accumulators live in registers (no aliasing through int8_t*), and the two heads give the
// FPU two independent dependency chains. Same float operations in the same order as v1,
// so v1 and v2 give bit-identical results. int_qk=1 additionally replaces the float Q.K
// with an integer dot of int8 q (per-head scale) and int8 k.
#define V8(p) (float)(p)[0], (float)(p)[1], (float)(p)[2], (float)(p)[3], (float)(p)[4], (float)(p)[5], (float)(p)[6], (float)(p)[7]

static HOT void softmax_inplace(float *a, int n) {
    float mx = a[0];
    for (int t = 1; t < n; t++) if (a[t] > mx) mx = a[t];
    float sum = 0.0f;
    for (int t = 0; t < n; t++) { a[t] = fast_expf(a[t] - mx); sum += a[t]; }
    float inv = 1.0f / sum;
    for (int t = 0; t < n; t++) a[t] = a[t] * inv;
}

static HOT void attn_pairs(int h0, int h1) {
    llm_state_t *s = G.s;
    const llm_cfg_t *c = &s->m->c;
    int kvd = c->kv, pos = G.pos, l = G.l, n = pos + 1, intqk = s->attn == 2;
    float *sa = s->att[h0 == 0 ? 0 : 1], *sb = sa + s->kv_seq;
    const int8_t *K = s->k8 + (size_t)l * s->kv_seq * kvd;
    const int8_t *V = s->v8 + (size_t)l * s->kv_seq * kvd;
    const float *ks = s->ks + (size_t)l * s->kv_seq, *vs = s->vs + (size_t)l * s->kv_seq;
    float isq = 1.0f / sqrtf(8.0f);
    for (int h = h0; h < h1; h += 2) {
        int kh = h >> 1;
        const float *qa = s->qkv + h * 8, *qb = qa + 8;
        if (!intqk) {
            // the FPU has 16 registers: do dims 0..3 for both heads, then 4..7 (same op order as v1)
            for (int half = 0; half < 2; half++) {
                int d = half * 4;
                float a0 = qa[d] * isq, a1 = qa[d + 1] * isq, a2 = qa[d + 2] * isq, a3 = qa[d + 3] * isq;
                float b0 = qb[d] * isq, b1 = qb[d + 1] * isq, b2 = qb[d + 2] * isq, b3 = qb[d + 3] * isq;
                const int8_t *kp = K + kh * 8 + d;
                for (int t = 0; t < n; t++, kp += kvd) {
                    float x = half ? sa[t] : 0.0f, y = half ? sb[t] : 0.0f, k;
                    k = kp[0]; x += k * a0; y += k * b0;
                    k = kp[1]; x += k * a1; y += k * b1;
                    k = kp[2]; x += k * a2; y += k * b2;
                    k = kp[3]; x += k * a3; y += k * b3;
                    if (half) { float kt = ks[t]; x = x * kt; y = y * kt; }
                    sa[t] = x; sb[t] = y;
                }
            }
        } else {
            int8_t qa8[8], qb8[8];
            float ca = q8_vec(qa8, qa, 8) * isq, cb = q8_vec(qb8, qb, 8) * isq;
            for (int hh = 0; hh < 2; hh++) {
                const int8_t *q8 = hh ? qb8 : qa8;
                float cc = hh ? cb : ca, *sc = hh ? sb : sa;
                int a0 = q8[0], a1 = q8[1], a2 = q8[2], a3 = q8[3], a4 = q8[4], a5 = q8[5], a6 = q8[6], a7 = q8[7];
                const int8_t *kp = K + kh * 8;
                for (int t = 0; t < n; t++, kp += kvd) {
                    int x = kp[0] * a0 + kp[1] * a1 + kp[2] * a2 + kp[3] * a3
                          + kp[4] * a4 + kp[5] * a5 + kp[6] * a6 + kp[7] * a7;
                    sc[t] = ((float)x * cc) * ks[t];
                }
            }
        }
        softmax_inplace(sa, n);
        softmax_inplace(sb, n);
        float *oa = s->att_out + h * 8, *ob = oa + 8;
        for (int half = 0; half < 2; half++) {
            int d = half * 4;
            float x0 = 0, x1 = 0, x2 = 0, x3 = 0, y0 = 0, y1 = 0, y2 = 0, y3 = 0;
            const int8_t *vp = V + kh * 8 + d;
            for (int t = 0; t < n; t++, vp += kvd) {
                float vt = vs[t], wa = sa[t] * vt, wb = sb[t] * vt, v;
                v = vp[0]; x0 += wa * v; y0 += wb * v;
                v = vp[1]; x1 += wa * v; y1 += wb * v;
                v = vp[2]; x2 += wa * v; y2 += wb * v;
                v = vp[3]; x3 += wa * v; y3 += wb * v;
            }
            oa[d] = x0; oa[d + 1] = x1; oa[d + 2] = x2; oa[d + 3] = x3;
            ob[d] = y0; ob[d + 1] = y1; ob[d + 2] = y2; ob[d + 3] = y3;
        }
    }
}

// ───────────────────────── forward ─────────────────────────

int llm_forward(llm_state_t *s, int tok, int pos, float *top1, float *margin) {
    const llm_model_t *m = s->m;
    const llm_cfg_t *c = &m->c;
    int D = c->dim, kvd = c->kv, Hp = c->hp;
    uint32_t t_start = cyc();
    float *x = s->x, *xb = s->xb;
    const int8_t *e = m->emb.q + (size_t)tok * D;
    float es = m->emb.s[tok];
    for (int j = 0; j < D; j++) x[j] = (float)e[j] * es;
    const float *cs = m->rope + (size_t)pos * c->hs;

    for (int l = 0; l < c->L; l++) {
        const llm_layer_t *ly = &m->layer[l];
        rmsnorm(xb, x, ly->rms_att, D);
        matmul(s, s->qkv, &ly->qkv, xb);                 // fused Q|K|V
        float *q = s->qkv, *k = s->qkv + D, *v = s->qkv + D + kvd;
        rope(q, c->nh, c->hs, cs);
        rope(k, c->nkv, c->hs, cs);
        size_t slot = (size_t)l * s->kv_seq + pos;
        s->ks[slot] = q8_vec(s->k8 + slot * kvd, k, kvd);
        s->vs[slot] = q8_vec(s->v8 + slot * kvd, v, kvd);

        uint32_t t0 = cyc();
        G.s = s; G.l = l; G.pos = pos;
        llm_job_fn af = (s->attn && c->hs == 8 && c->nh == 2 * c->nkv && c->nh % 4 == 0) ? attn_pairs : attn_heads;
        if (s->ncores > 1 && llm_parallel) llm_parallel(af, c->nh);
        else af(0, c->nh);
        s->cyc_att += cyc() - t0;

        matmul(s, xb, &ly->wo, s->att_out);
        for (int j = 0; j < D; j++) x[j] = x[j] + xb[j];

        rmsnorm(xb, x, ly->rms_ffn, D);
        matmul(s, s->h13, &ly->w13, xb);                 // fused W1|W3
        for (int j = 0; j < Hp; j++) {
            float h1 = s->h13[j], h3 = s->h13[Hp + j];
            s->hh[j] = (h1 / (1.0f + fast_expf(-h1))) * h3;
        }
        matmul(s, xb, &ly->w2, s->hh);
        for (int j = 0; j < D; j++) x[j] = x[j] + xb[j];
    }
    rmsnorm(xb, x, m->rms_final, D);
    matmul(s, s->logits, &m->emb, xb);                    // tied classifier

    int best = 0;
    float b1 = s->logits[0], b2 = -1e30f;
    for (int i = 1; i < c->vocab; i++) {
        float v = s->logits[i];
        if (v > b1) { b2 = b1; b1 = v; best = i; }
        else if (v > b2) b2 = v;
    }
    if (top1) *top1 = b1;
    if (margin) *margin = b1 - b2;
    s->cyc_all += cyc() - t_start;
    return best;
}
