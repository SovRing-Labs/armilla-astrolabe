/* ARMILLA-FAST-0 speed ladder (2026-10-13): armilla_bitnet.c + ABN_FAST levels (1 threaded precompute, 2 +2-bit packed codes), same logits hash as the baseline.
 * Armilla BitNet b1.58 2B4T whole-model greedy decode (OUTREACH-P1-DECODE, 2026-10-13).
 * Every ternary linear (q k v o gate up down, 30 layers) runs on the Armilla AVX2 pull (experiments/avx2/pull_avx2.c form):
 * int8 activation, 2-trit codes, 16-byte answer tables pulled with pshufb over rings 251/241/239, lazy 16-bit sums with vector
 * Barrett, one CRT (Garner-form) read-out to the exact signed integer. Everything else (RMSNorm, RoPE, softmax, relu2, logits)
 * is IEEE basic ops in a fixed order: no FMA, no libm transcendental (own exp/sin/cos in double), so the run is bit-reproducible
 * across x86 CPUs. Output: token ids + SHA-256 of the logits stream is computed by the caller from --dump-logits.
 * Build: gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp armilla_bitnet.c -o armilla_bitnet -lm
 * Use:   armilla_bitnet MODEL.safetensors NGEN THREADS tok0 tok1 ... [env ABN_SCALE_DIV=1 ABN_LOGITS=file] */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>

#define D 2560
#define FF 6912
#define NL 30
#define NH 20
#define NKV 5
#define HD 128
#define V 128256
#define MAXCTX 512
#define R 3
static const int MODS[R] = {251, 241, 239};

/* ---------- Armilla pull (from experiments/avx2/pull_avx2.c) ---------- */
static uint8_t MODT[R][512];            /* MODT[r][s + 256] = s mod MODS[r] for s in [-256, 255] */
static int FAST;
static void pull2_precompute(const int8_t *x, int K, uint8_t *lut) {
    int G2 = K / 2;
    if (FAST >= 1) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < G2; j++) {
            int a = x[2 * j], b = x[2 * j + 1], s[9];
            for (int p = 0; p < 9; p++) s[p] = (p % 3 - 1) * a + (p / 3 - 1) * b + 256;
            for (int r = 0; r < R; r++) {
                uint8_t t[32] = {0};
                for (int p = 0; p < 9; p++) t[p] = t[16 + p] = MODT[r][s[p]];
                memcpy(lut + ((size_t)j * R + r) * 32, t, 32);
            }
        }
        return;
    }
    for (int j = 0; j < G2; j++) {
        int a = x[2 * j], b = x[2 * j + 1], s[9];
        for (int p = 0; p < 9; p++) s[p] = (p % 3 - 1) * a + (p / 3 - 1) * b;
        for (int r = 0; r < R; r++) {
            uint8_t *t = lut + ((size_t)j * R + r) * 32;
            for (int p = 0; p < 16; p++) {
                int v = 0;
                if (p < 9) { v = s[p] % MODS[r]; if (v < 0) v += MODS[r]; }
                t[p] = t[16 + p] = (uint8_t)v;
            }
        }
    }
}
static inline __m256i barrett16(__m256i a, int m) {
    __m256i q = _mm256_mulhi_epu16(a, _mm256_set1_epi16((short)(65536 / m)));
    __m256i r = _mm256_sub_epi16(a, _mm256_mullo_epi16(q, _mm256_set1_epi16((short)m)));
    __m256i mm = _mm256_set1_epi16((short)m);
    r = _mm256_min_epu16(r, _mm256_sub_epi16(r, mm));
    return _mm256_min_epu16(r, _mm256_sub_epi16(r, mm));
}
/* blocks [b0,b1): j-tiles of 128 groups (one 12 KB table tile in L1) swept over every block of the range; codes blocked [ob/32][j][32] */
static void pull2_range(const uint8_t *codes, const uint8_t *lut, int G2, int O, int b0, int b1, uint8_t *out, __m256i *acc) {
    int nb = b1 - b0;
    for (int i = 0; i < nb * R * 2; i++) acc[i] = _mm256_setzero_si256();
    for (int j0 = 0; j0 < G2; j0 += 128) {
        int j1 = j0 + 128 < G2 ? j0 + 128 : G2;
        for (int b = 0; b < nb; b++) {
            const uint8_t *cb = codes + ((size_t)(b0 + b) * G2 + j0) * 32;
            __m256i lo0 = acc[b * 6 + 0], hi0 = acc[b * 6 + 1], lo1 = acc[b * 6 + 2], hi1 = acc[b * 6 + 3], lo2 = acc[b * 6 + 4], hi2 = acc[b * 6 + 5];
            for (int j = j0; j < j1; j++, cb += 32) {
                __m256i idx = _mm256_load_si256((const void *)cb);
                const uint8_t *T = lut + (size_t)j * R * 32;
                __m256i v0 = _mm256_shuffle_epi8(_mm256_load_si256((const void *)T), idx);
                __m256i v1 = _mm256_shuffle_epi8(_mm256_load_si256((const void *)(T + 32)), idx);
                __m256i v2 = _mm256_shuffle_epi8(_mm256_load_si256((const void *)(T + 64)), idx);
                lo0 = _mm256_add_epi16(lo0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v0))); hi0 = _mm256_add_epi16(hi0, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v0, 1)));
                lo1 = _mm256_add_epi16(lo1, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v1))); hi1 = _mm256_add_epi16(hi1, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v1, 1)));
                lo2 = _mm256_add_epi16(lo2, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v2))); hi2 = _mm256_add_epi16(hi2, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v2, 1)));
            }
            acc[b * 6 + 0] = barrett16(lo0, MODS[0]); acc[b * 6 + 1] = barrett16(hi0, MODS[0]);
            acc[b * 6 + 2] = barrett16(lo1, MODS[1]); acc[b * 6 + 3] = barrett16(hi1, MODS[1]);
            acc[b * 6 + 4] = barrett16(lo2, MODS[2]); acc[b * 6 + 5] = barrett16(hi2, MODS[2]);
        }
    }
    for (int b = 0; b < nb; b++)
        for (int r = 0; r < R; r++) {
            uint16_t a[16], c[16];
            _mm256_storeu_si256((void *)a, acc[b * 6 + 2 * r]); _mm256_storeu_si256((void *)c, acc[b * 6 + 2 * r + 1]);
            for (int l = 0; l < 16; l++) { out[(size_t)r * O + (b0 + b) * 32 + l] = (uint8_t)a[l]; out[(size_t)r * O + (b0 + b) * 32 + 16 + l] = (uint8_t)c[l]; }
        }
}
/* L2: two groups per code byte (low nibble = group j0+jj, high nibble = group j0+64+jj within a 128-group tile): 2 bits per weight.
 * codes2 blocked [ob/32][G2/2][32]. Same table math as pull2_range, so the residues are bit-identical. */
#define PULL_GROUP(IDX, J) do { const uint8_t *T = lut + (size_t)(J) * R * 32; \
    __m256i v0 = _mm256_shuffle_epi8(_mm256_load_si256((const void *)T), IDX); \
    __m256i v1 = _mm256_shuffle_epi8(_mm256_load_si256((const void *)(T + 32)), IDX); \
    __m256i v2 = _mm256_shuffle_epi8(_mm256_load_si256((const void *)(T + 64)), IDX); \
    lo0 = _mm256_add_epi16(lo0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v0))); hi0 = _mm256_add_epi16(hi0, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v0, 1))); \
    lo1 = _mm256_add_epi16(lo1, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v1))); hi1 = _mm256_add_epi16(hi1, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v1, 1))); \
    lo2 = _mm256_add_epi16(lo2, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v2))); hi2 = _mm256_add_epi16(hi2, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v2, 1))); } while (0)
static void pull2_range_packed(const uint8_t *codes2, const uint8_t *lut, int G2, int O, int b0, int b1, uint8_t *out, __m256i *acc) {
    int nb = b1 - b0; const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (int i = 0; i < nb * R * 2; i++) acc[i] = _mm256_setzero_si256();
    for (int j0 = 0; j0 < G2; j0 += 128) {
        for (int b = 0; b < nb; b++) {
            const uint8_t *cb = codes2 + ((size_t)(b0 + b) * (G2 / 2) + j0 / 2) * 32;
            __m256i lo0 = acc[b * 6 + 0], hi0 = acc[b * 6 + 1], lo1 = acc[b * 6 + 2], hi1 = acc[b * 6 + 3], lo2 = acc[b * 6 + 4], hi2 = acc[b * 6 + 5];
            for (int jj = 0; jj < 64; jj++, cb += 32) {
                __m256i v = _mm256_load_si256((const void *)cb);
                __m256i i0 = _mm256_and_si256(v, m4), i1 = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
                PULL_GROUP(i0, j0 + jj); PULL_GROUP(i1, j0 + 64 + jj);
            }
            acc[b * 6 + 0] = barrett16(lo0, MODS[0]); acc[b * 6 + 1] = barrett16(hi0, MODS[0]);
            acc[b * 6 + 2] = barrett16(lo1, MODS[1]); acc[b * 6 + 3] = barrett16(hi1, MODS[1]);
            acc[b * 6 + 4] = barrett16(lo2, MODS[2]); acc[b * 6 + 5] = barrett16(hi2, MODS[2]);
        }
    }
    for (int b = 0; b < nb; b++)
        for (int r = 0; r < R; r++) {
            uint16_t a[16], c[16];
            _mm256_storeu_si256((void *)a, acc[b * 6 + 2 * r]); _mm256_storeu_si256((void *)c, acc[b * 6 + 2 * r + 1]);
            for (int l = 0; l < 16; l++) { out[(size_t)r * O + (b0 + b) * 32 + l] = (uint8_t)a[l]; out[(size_t)r * O + (b0 + b) * 32 + 16 + l] = (uint8_t)c[l]; }
        }
}
/* CRT read-out: residues -> exact signed integer */
static int64_t CRTC[R]; enum { CRTM = 251 * 241 * 239 };
static int64_t inv_mod(int64_t a, int64_t m) { for (int64_t x = 1; x < m; x++) if ((a * x) % m == 1) return x; return 0; }
static void crt_init(void) {
    for (int r = 0; r < R; r++) for (int v = -256; v < 256; v++) { int m = v % MODS[r]; if (m < 0) m += MODS[r]; MODT[r][v + 256] = (uint8_t)m; }
    for (int r = 0; r < R; r++) { int64_t Mi = CRTM / MODS[r]; CRTC[r] = (Mi * inv_mod(Mi % MODS[r], MODS[r])) % CRTM; }
}
static inline int32_t crt_read(const uint8_t *res, int O, int o) {
    int64_t v = 0; for (int r = 0; r < R; r++) v += (int64_t)res[(size_t)r * O + o] * CRTC[r];
    v %= CRTM; if (v > CRTM / 2) v -= CRTM; return (int32_t)v;
}

/* ---------- model ---------- */
#include <sys/mman.h>
static void *big_alloc(size_t n) {   /* L6: 2 MB aligned + MADV_HUGEPAGE (THP is 'madvise' on the pod) */
    size_t a = (size_t)2 << 20; n = (n + a - 1) / a * a; void *p = aligned_alloc(a, n); if (p && FAST >= 6) madvise(p, n, MADV_HUGEPAGE); return p;
}
typedef struct { uint8_t *codes, *codes2, *w3; int K, O; float wscale; } Lin;
typedef struct { Lin q, k, v, o, g, u, d; const uint16_t *in_ln, *post_ln, *attn_sub, *ffn_sub; } Layer;
static Layer L[NL];
static const uint16_t *EMB, *FNORM;
static const uint8_t *base; static size_t hdrlen; static char *hdr;
static int F16KV, F16EMB, SCALE_DIV, ROPE_IL, SCALE_MODE, ACT_SILU;

static inline float f16r(float x) { return _cvtsh_ss(_cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT)); }
static float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static const uint8_t *tensor(const char *name, size_t *nbytes) {
    char key[256]; snprintf(key, sizeof key, "\"%s\"", name);
    char *p = strstr(hdr, key); if (!p) { fprintf(stderr, "missing %s\n", name); exit(2); }
    p = strstr(p, "data_offsets"); p = strchr(p, '['); size_t a, b; sscanf(p + 1, "%zu,%zu", &a, &b);
    if (nbytes) *nbytes = b - a; return base + 8 + hdrlen + a;
}
/* HF packing: packed[O/4][K], 2-bit field i of a byte -> output row i*(O/4)+r, value = field - 1 */
static void pack_codes(Lin *l);
static void load_lin(Lin *l, const char *pre, int O, int K) {
    char n[256]; snprintf(n, sizeof n, "%s.weight", pre); const uint8_t *pk = tensor(n, NULL);
    snprintf(n, sizeof n, "%s.weight_scale", pre); l->wscale = bf(*(const uint16_t *)tensor(n, NULL));
    l->K = K; l->O = O; l->codes = aligned_alloc(64, (size_t)K / 2 * O);
    int O4 = O / 4;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < K / 2; j++)
        for (int f = 0; f < 4; f++)
            for (int r = 0; r < O4; r++) {
                int t0 = ((pk[(size_t)r * K + 2 * j] >> (2 * f)) & 3) - 1, t1 = ((pk[(size_t)r * K + 2 * j + 1] >> (2 * f)) & 3) - 1;
                { int o = f * O4 + r; l->codes[((size_t)(o >> 5) * (K / 2) + j) * 32 + (o & 31)] = (uint8_t)((t0 + 1) + 3 * (t1 + 1)); }
            }
    if (FAST >= 3) {   /* plane 3: per output row, 128-k blocks of 32 bytes; byte i field ff = weight k=blk*128+ff*32+i, stored as raw field (0,1,2 = -1,0,+1) */
        l->w3 = big_alloc((size_t)O * K / 4);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            int r = o % O4, f = o / O4; const uint8_t *src = pk + (size_t)r * K; uint8_t *dst = l->w3 + (size_t)o * (K / 4);
            for (int blk = 0; blk < K / 128; blk++) for (int i = 0; i < 32; i++) {
                uint8_t b = 0; for (int ff = 0; ff < 4; ff++) b |= (uint8_t)(((src[blk * 128 + ff * 32 + i] >> (2 * f)) & 3) << (2 * ff));
                dst[blk * 32 + i] = b;
            }
        }
    }
    if (FAST == 2) pack_codes(l);
    if (FAST >= 3 && !getenv("ABN_CHECK")) { free(l->codes); l->codes = NULL; }
}
static void pack_codes(Lin *l) {            /* codes (one byte per group) -> codes2 (two groups per byte, tile-paired) */
    int G2 = l->K / 2, nb = l->O / 32; l->codes2 = aligned_alloc(64, (size_t)G2 / 2 * l->O);
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < nb; b++)
        for (int j0 = 0; j0 < G2; j0 += 128)
            for (int jj = 0; jj < 64; jj++)
                for (int ln = 0; ln < 32; ln++)
                    l->codes2[((size_t)b * (G2 / 2) + j0 / 2 + jj) * 32 + ln] = l->codes[((size_t)b * G2 + j0 + jj) * 32 + ln] | (l->codes[((size_t)b * G2 + j0 + 64 + jj) * 32 + ln] << 4);
    if (!getenv("ABN_CHECK") && FAST < 3) { free(l->codes); l->codes = NULL; }
}
static void load_model(const char *path) {
    int fd = open(path, O_RDONLY); struct stat st; fstat(fd, &st);
    base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t hl; memcpy(&hl, base, 8); hdrlen = hl; hdr = malloc(hl + 1); memcpy(hdr, base + 8, hl); hdr[hl] = 0;
    EMB = (const uint16_t *)tensor("model.embed_tokens.weight", NULL); FNORM = (const uint16_t *)tensor("model.norm.weight", NULL);
    for (int i = 0; i < NL; i++) {
        char n[200]; Layer *y = &L[i];
        #define LN(field, fmt) snprintf(n, sizeof n, "model.layers.%d." fmt, i); y->field = (const uint16_t *)tensor(n, NULL);
        LN(in_ln, "input_layernorm.weight") LN(post_ln, "post_attention_layernorm.weight")
        LN(attn_sub, "self_attn.attn_sub_norm.weight") LN(ffn_sub, "mlp.ffn_sub_norm.weight")
        #define LL(field, fmt, O, K) snprintf(n, sizeof n, "model.layers.%d." fmt, i); load_lin(&y->field, n, O, K);
        LL(q, "self_attn.q_proj", NH * HD, D) LL(k, "self_attn.k_proj", NKV * HD, D) LL(v, "self_attn.v_proj", NKV * HD, D)
        LL(o, "self_attn.o_proj", D, D) LL(g, "mlp.gate_proj", FF, D) LL(u, "mlp.up_proj", FF, D) LL(d, "mlp.down_proj", D, FF)
    }
}

/* ---------- deterministic math (basic IEEE ops only) ---------- */
static double dexp(double x) {            /* |x| < 700 */
    const double LN2 = 0.6931471805599453, INV = 1.4426950408889634;
    double kf = x * INV; long k = (long)(kf < 0 ? kf - 0.5 : kf + 0.5);
    double r = x - (double)k * LN2, t = 1.0, s = 1.0;
    for (int i = 1; i <= 20; i++) { t = t * r / (double)i; s += t; }
    return ldexp(s, (int)k);               /* ldexp is exact scaling */
}
static void dsincos(double a, double *s, double *c) {   /* a in [0, ~1000] */
    const double PI2 = 6.283185307179586, PI2L = 2.4492935982947064e-16;
    long k = (long)(a / PI2); double r = (a - (double)k * PI2) - (double)k * PI2L;
    if (r > 3.141592653589793) r -= PI2;
    double r2 = r * r, ts = r, tc = 1.0, ss = r, sc = 1.0;
    for (int i = 1; i <= 14; i++) { tc = -tc * r2 / ((2.0 * i - 1) * (2.0 * i)); sc += tc; ts = -ts * r2 / ((2.0 * i) * (2.0 * i + 1)); ss += ts; }
    *s = ss; *c = sc;
}

static void rmsnorm(const float *x, const uint16_t *w, float *out, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float inv = (float)(1.0 / sqrt(ss / n + 1e-5));
    for (int i = 0; i < n; i++) out[i] = (x[i] * inv) * bf(w[i]);
}
static float quant_act(const float *x, int n, int8_t *q) {   /* per-token absmax int8 */
    float mx = 0; for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    if (mx < 1e-5f) mx = 1e-5f;
    float sc = 127.0f / mx;
    for (int i = 0; i < n; i++) { float v = nearbyintf(x[i] * sc); if (v > 127) v = 127; if (v < -128) v = -128; q[i] = (int8_t)v; }
    return mx / 127.0f;                    /* dequant step */
}
static uint8_t *LUT, *RES; static const int8_t *LUTX;
static int8_t *XQ;
/* y = (W x_q) * act_step * wscale (or / wscale) */
static double T_PRE, T_PULL, T_LOG, T_CRT, T_FWD;
static double now(void);
/* FLOW LEDGER (ABN_LEDGER=file.jsonl): per decode token, per stage {pull, head1, head2}: wall s, bytes read, per-thread busy cycles (rdtsc inside the omp region);
 * barrier wait = wall - busy. Counters only; off => the loops below are the same loops (hash unchanged). Idea: per-ring counters of the traffic (VOCAB "outer rings counting"). */
enum { LS_PULL, LS_HEAD1, LS_HEAD2, LS_N }; static int LEDGER; static double LBUSY[LS_N][64], LWALL[LS_N], LBYTES[LS_N]; static double TSCHZ; static FILE *LF;
#define LBEG uint64_t _t0 = LEDGER ? __rdtsc() : 0;
#define LEND(S) if (LEDGER) LBUSY[S][omp_get_thread_num()] += (double)(__rdtsc() - _t0);
static int XSUM;
/* plane 3: exact ternary dot, 4 output rows per pass. u in {0,1,2} (= w+1) x int8 x via maddubs, int16 sums of 4 fields <= 2048, madd to int32;
 * dot = sum(u*x) - sum(x). No tables, no residues, no CRT. Idea of the u8*s8 maddubs ternary dot as in llama.cpp I2_S / bitnet.cpp (cited, re-derived). */
static void plane3(const Lin *l, const int8_t *x, int32_t *out) {
    const int K = l->K, O = l->O, KB = K / 128; const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    #pragma omp parallel
    { LBEG
    #pragma omp for schedule(static) nowait
    for (int o0 = 0; o0 < O; o0 += 4) {
        __m256i a[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
        const uint8_t *w[4]; for (int q = 0; q < 4; q++) w[q] = l->w3 + (size_t)(o0 + q) * (K / 4);
        for (int b = 0; b < KB; b++) {
            const __m256i x0 = _mm256_loadu_si256((const __m256i *)(x + b * 128)), x1 = _mm256_loadu_si256((const __m256i *)(x + b * 128 + 32)),
                          x2 = _mm256_loadu_si256((const __m256i *)(x + b * 128 + 64)), x3 = _mm256_loadu_si256((const __m256i *)(x + b * 128 + 96));
            for (int q = 0; q < 4; q++) {
                __m256i v = _mm256_load_si256((const __m256i *)(w[q] + b * 32));
                __m256i s = _mm256_maddubs_epi16(_mm256_and_si256(v, m3), x0);
                s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v, 2), m3), x1));
                s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v, 4), m3), x2));
                s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(v, 6), m3), x3));
                a[q] = _mm256_add_epi32(a[q], _mm256_madd_epi16(s, ones));
            }
        }
        for (int q = 0; q < 4; q++) {
            __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a[q]), _mm256_extracti128_si256(a[q], 1));
            h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0x4E)); h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0xB1));
            out[o0 + q] = _mm_cvtsi128_si32(h) - XSUM;
        }
    }
    LEND(LS_PULL) }
}
static void lin_pre(const int8_t *xq, int K) { double t = now(); LUTX = xq;
    if (FAST >= 3) { int sx = 0; for (int i = 0; i < K; i++) sx += xq[i]; XSUM = sx; T_PRE += now() - t; return; }
    pull2_precompute(xq, K, LUT); T_PRE += now() - t; }
static void lin_run(const Lin *l, float step, float *y) {
    int O = l->O, nb = O / 32;
    if (FAST >= 3) {
        static int32_t *dots; if (!dots) dots = aligned_alloc(64, 8192 * 4);
        double t0 = now(); plane3(l, LUTX, dots); double dt = now() - t0; T_PULL += dt; LWALL[LS_PULL] += dt; LBYTES[LS_PULL] += (double)O * l->K / 4;
        float f = SCALE_MODE == 2 ? step : SCALE_DIV ? step / l->wscale : step * l->wscale;
        for (int o = 0; o < O; o++) y[o] = (float)dots[o] * f;
        return;
    }
    double t0 = now();
    #pragma omp parallel
    {
        int nt = omp_get_num_threads(), t = omp_get_thread_num(), b0 = nb * t / nt, b1 = nb * (t + 1) / nt;
        __m256i acc[(1728 / 4 + 8) * 6];   /* max blocks per thread: 216 blocks / 1 thread */
        if (b1 - b0 > 0) { if (FAST >= 2) pull2_range_packed(l->codes2, LUT, l->K / 2, O, b0, b1, RES, acc); else pull2_range(l->codes, LUT, l->K / 2, O, b0, b1, RES, acc); }
    }
    T_PULL += now() - t0;
    float f = SCALE_MODE == 2 ? step : SCALE_DIV ? step / l->wscale : step * l->wscale;
    double tc = now();
    if (FAST >= 3) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) y[o] = (float)crt_read(RES, O, o) * f;
    } else for (int o = 0; o < O; o++) y[o] = (float)crt_read(RES, O, o) * f;
    T_CRT += now() - tc;
    if (getenv("ABN_CHECK")) {            /* direct integer dot from the codes: the pull must equal it exactly */
        int bad = 0, G2 = l->K / 2;
        for (int o = 0; o < O; o++) {
            int64_t d = 0; for (int j = 0; j < G2; j++) { int c = l->codes[((size_t)(o >> 5) * G2 + j) * 32 + (o & 31)]; d += (c % 3 - 1) * LUTX[2 * j] + (c / 3 - 1) * LUTX[2 * j + 1]; }
            if (d != crt_read(RES, O, o)) bad++;
        }
        fprintf(stderr, "check O=%d K=%d mismatches=%d\n", O, l->K, bad);
    }
}

static float *KC[NL], *VC[NL];
static double INVF[HD / 2];
static void rope(float *x, int nh, int pos) {
    for (int i = 0; i < HD / 2; i++) {
        double s, c; dsincos((double)pos * INVF[i], &s, &c);
        for (int h = 0; h < nh; h++) {
            float *p = x + h * HD; int i0 = ROPE_IL ? 2 * i : i, i1 = ROPE_IL ? 2 * i + 1 : i + HD / 2; float a = p[i0], b = p[i1];
            p[i0] = (float)((double)a * c - (double)b * s); p[i1] = (float)((double)b * c + (double)a * s);
        }
    }
}

static double RSN[HD / 2], RCS[HD / 2];   /* L5: sin/cos of pos*INVF computed once per token, not once per layer x {q,k} */
static void rope_tab(int pos) { for (int i = 0; i < HD / 2; i++) dsincos((double)pos * INVF[i], &RSN[i], &RCS[i]); }
static void rope5(float *x, int nh) {
    for (int i = 0; i < HD / 2; i++) {
        double s = RSN[i], c = RCS[i];
        for (int h = 0; h < nh; h++) {
            float *p = x + h * HD; int i0 = ROPE_IL ? 2 * i : i, i1 = ROPE_IL ? 2 * i + 1 : i + HD / 2; float a = p[i0], b = p[i1];
            p[i0] = (float)((double)a * c - (double)b * s); p[i1] = (float)((double)b * c + (double)a * s);
        }
    }
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
/* STEP 4: certified int8 head. Pass 1 reads the int8 copy of the tied embedding (half the bytes), exact int32 dot with the int8 activation.
 * true_v - approx_v = R_v.x + Q_v.d  (E=Q+R row split, x = s_x(xq)+d), so |true-approx| <= |R_v||x| + |Q_v||d|  (Cauchy-Schwarz; both norms measured, not assumed).
 * Survivors = rows whose upper bound reaches the best lower bound; only they are re-scored with the exact bf16 float dot of the baseline, so the argmax is the
 * baseline's argmax by construction. Non-survivors get -1e30 in `logits`. */
static int8_t *EQ; static float *SV, *QN, *RN; static int *SURV; static long N_SURV, N_STEPS, MAX_SURV; static float *APX;
static void head_init(void) {
    EQ = big_alloc((size_t)V * D); SV = malloc(V * 4); QN = malloc(V * 4); RN = malloc(V * 4); SURV = malloc(V * 4); APX = malloc(V * 4);
    #pragma omp parallel for schedule(static)
    for (int vi = 0; vi < V; vi++) {
        const uint16_t *w = EMB + (size_t)vi * D; float mx = 0; for (int i = 0; i < D; i++) { float a = fabsf(bf(w[i])); if (a > mx) mx = a; }
        float sv = mx > 0 ? mx / 127.0f : 1.0f; double qn = 0, rn = 0;
        for (int i = 0; i < D; i++) { float e = bf(w[i]); int q = (int)nearbyintf(e / sv); if (q > 127) q = 127; if (q < -127) q = -127; EQ[(size_t)vi * D + i] = (int8_t)q; double r = (double)e - (double)q * (double)sv; rn += r * r; qn += (double)q * q * (double)sv * (double)sv; }
        SV[vi] = sv; QN[vi] = (float)sqrt(qn); RN[vi] = (float)sqrt(rn);
    }
}
static void head_cert(const float *hn, float *logits) {
    if (getenv("ABN_DUMPHN")) { static FILE *df; if (!df) df = fopen(getenv("ABN_DUMPHN"), "wb"); fwrite(hn, 4, D, df); fflush(df); }
    static int8_t xh[D]; float mx = 0; double nx = 0; for (int i = 0; i < D; i++) { float a = fabsf(hn[i]); if (a > mx) mx = a; nx += (double)hn[i] * hn[i]; }
    float sx = mx > 0 ? mx / 127.0f : 1.0f; double rho = 0;
    for (int i = 0; i < D; i++) { int q = (int)nearbyintf(hn[i] / sx); if (q > 127) q = 127; if (q < -127) q = -127; xh[i] = (int8_t)q; double d = (double)hn[i] - (double)q * (double)sx; rho += d * d; }
    float nxs = (float)sqrt(nx), rhos = (float)sqrt(rho); const __m256i ones = _mm256_set1_epi16(1);
    double tp1 = now();
    #pragma omp parallel
    { LBEG
    #pragma omp for schedule(static) nowait
    for (int vi = 0; vi < V; vi++) {
        const int8_t *w = EQ + (size_t)vi * D; __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        for (int i = 0; i < D; i += 64) {
            __m256i q0 = _mm256_load_si256((const __m256i *)(w + i)), q1 = _mm256_load_si256((const __m256i *)(w + i + 32));
            __m256i x0 = _mm256_loadu_si256((const __m256i *)(xh + i)), x1 = _mm256_loadu_si256((const __m256i *)(xh + i + 32));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(q0), _mm256_sign_epi8(x0, q0)), ones));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(q1), _mm256_sign_epi8(x1, q1)), ones));
        }
        __m256i a = _mm256_add_epi32(a0, a1); __m128i hh = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        hh = _mm_add_epi32(hh, _mm_shuffle_epi32(hh, 0x4E)); hh = _mm_add_epi32(hh, _mm_shuffle_epi32(hh, 0xB1));
        APX[vi] = (float)_mm_cvtsi128_si32(hh) * SV[vi] * sx;
    }
    LEND(LS_HEAD1) }
    LWALL[LS_HEAD1] += now() - tp1; LBYTES[LS_HEAD1] += (double)V * D;
    float lower = -3.0e38f; const float slack = 1e-5f * nxs;
    for (int vi = 0; vi < V; vi++) { float bd = RN[vi] * nxs + QN[vi] * rhos + slack * (QN[vi] + RN[vi]); float lo = APX[vi] - bd; if (lo > lower) lower = lo; }
    int ns = 0; for (int vi = 0; vi < V; vi++) { float bd = RN[vi] * nxs + QN[vi] * rhos + slack * (QN[vi] + RN[vi]); logits[vi] = -1e30f; if (APX[vi] + bd >= lower) SURV[ns++] = vi; }
    double tp2 = now();
    #pragma omp parallel
    { LBEG
    #pragma omp for schedule(static) nowait
    for (int si = 0; si < ns; si++) {
        int vi = SURV[si]; const uint16_t *w = EMB + (size_t)vi * D; __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < D; i += 8) {
            __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const void *)(w + i))), 16));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wf, _mm256_loadu_ps(hn + i)));
        }
        float a[8]; _mm256_storeu_ps(a, acc);
        logits[vi] = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
    }
    LEND(LS_HEAD2) }
    LWALL[LS_HEAD2] += now() - tp2; LBYTES[LS_HEAD2] += (double)ns * D * 2;
    if (getenv("ABN_HEADCHECK")) {   /* every step: certified argmax == argmax of the full baseline bf16 logits */
        int bc = 0, bf_ = 0; float bcv = -3e38f, bfv = -3e38f;
        for (int vi = 0; vi < V; vi++) {
            const uint16_t *w = EMB + (size_t)vi * D; __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < D; i += 8) { __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const void *)(w + i))), 16)); acc = _mm256_add_ps(acc, _mm256_mul_ps(wf, _mm256_loadu_ps(hn + i))); }
            float a[8]; _mm256_storeu_ps(a, acc); float t = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
            if (t > bfv) { bfv = t; bf_ = vi; } if (logits[vi] > bcv) { bcv = logits[vi]; bc = vi; }
        }
        static long chk; chk++; if (bc != bf_) { fprintf(stderr, "HEADCHECK FAIL step %ld cert=%d full=%d\n", chk, bc, bf_); exit(9); }
        if (chk % 16 == 0 || chk == 1) fprintf(stderr, "headcheck ok through step %ld\n", chk);
    }
    N_SURV += ns; N_STEPS++; if (ns > MAX_SURV) MAX_SURV = ns;
}
static void ledger_emit(int tok, double wall, int nt) {
    static const char *nm[LS_N] = {"pull", "head1", "head2"}; static char prev[17] = "0000000000000000";
    char row[2048]; int n = snprintf(row, sizeof row, "{\"tok\":%d,\"wall_s\":%.6f,\"threads\":%d,\"stages\":{", tok, wall, nt); double sw = 0;
    for (int st = 0; st < LS_N; st++) {
        n += snprintf(row + n, sizeof row - n, "%s\"%s\":{\"wall_s\":%.6f,\"bytes\":%.0f,\"busy_s\":[", st ? "," : "", nm[st], LWALL[st], LBYTES[st]); sw += LWALL[st];
        for (int t = 0; t < nt; t++) n += snprintf(row + n, sizeof row - n, "%s%.6f", t ? "," : "", LBUSY[st][t] / TSCHZ);
        n += snprintf(row + n, sizeof row - n, "]}");
    }
    n += snprintf(row + n, sizeof row - n, "},\"other_s\":%.6f", wall - sw);
    /* hash chain (FNV-1a 64 over prev|row) so a run's ledger is a verifiable record */
    uint64_t hsh = 1469598103934665603ULL; for (const char *c = prev; *c; c++) hsh = (hsh ^ (uint8_t)*c) * 1099511628211ULL; for (int i = 0; i < n; i++) hsh = (hsh ^ (uint8_t)row[i]) * 1099511628211ULL;
    snprintf(prev, sizeof prev, "%016llx", (unsigned long long)hsh); fprintf(LF, "%s,\"chain\":\"%s\"}\n", row, prev);
}
static void forward(int tok, int pos, float *h, float *logits, int want_logits) {
    static float xn[FF], q[NH * HD], k[NKV * HD], v[NKV * HD], att[D], tmp[D], g[FF], u[FF], sub[FF];
    for (int i = 0; i < D; i++) h[i] = F16EMB ? f16r(bf(EMB[(size_t)tok * D + i])) : bf(EMB[(size_t)tok * D + i]);
    if (FAST >= 5) rope_tab(pos);
    for (int li = 0; li < NL; li++) {
        Layer *y = &L[li]; float step;
        rmsnorm(h, y->in_ln, xn, D); step = quant_act(xn, D, XQ); lin_pre(XQ, D);
        lin_run(&y->q, step, q); lin_run(&y->k, step, k); lin_run(&y->v, step, v);
        if (FAST >= 5) { rope5(q, NH); rope5(k, NKV); } else { rope(q, NH, pos); rope(k, NKV, pos); }
        if (F16KV) for (int i = 0; i < NKV * HD; i++) { k[i] = f16r(k[i]); v[i] = f16r(v[i]); }
        memcpy(KC[li] + (size_t)pos * NKV * HD, k, sizeof k); memcpy(VC[li] + (size_t)pos * NKV * HD, v, sizeof v);
        #pragma omp parallel for schedule(static) if (FAST >= 5)
        for (int hh = 0; hh < NH; hh++) {
            float sc[MAXCTX]; int kv = hh / (NH / NKV); const float *qh = q + hh * HD; float mx = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const float *kh = KC[li] + ((size_t)t * NKV + kv) * HD; double s = 0;
                for (int i = 0; i < HD; i++) s += (double)qh[i] * (double)kh[i];
                sc[t] = (float)(s / 11.313708498984761);   /* sqrt(128) */
                if (sc[t] > mx) mx = sc[t];
            }
            double den = 0; for (int t = 0; t <= pos; t++) { double e = dexp((double)sc[t] - (double)mx); sc[t] = (float)e; den += e; }
            for (int i = 0; i < HD; i++) {
                double a = 0; for (int t = 0; t <= pos; t++) a += (double)sc[t] * (double)VC[li][((size_t)t * NKV + kv) * HD + i];
                att[hh * HD + i] = (float)(a / den);
            }
        }
        rmsnorm(att, y->attn_sub, sub, D); step = quant_act(sub, D, XQ); lin_pre(XQ, D); lin_run(&y->o, step, tmp);
        for (int i = 0; i < D; i++) h[i] += tmp[i];
        rmsnorm(h, y->post_ln, xn, D); step = quant_act(xn, D, XQ); lin_pre(XQ, D);
        lin_run(&y->g, step, g); lin_run(&y->u, step, u);
        for (int i = 0; i < FF; i++) {
            if (ACT_SILU) g[i] = (float)((double)g[i] / (1.0 + dexp(-(double)g[i]))) * u[i];
            else { float a = g[i] > 0 ? g[i] : 0; g[i] = a * a * u[i]; }
        }
        rmsnorm(g, y->ffn_sub, sub, FF); step = quant_act(sub, FF, XQ); lin_pre(XQ, FF); lin_run(&y->d, step, tmp);
        for (int i = 0; i < D; i++) h[i] += tmp[i];
    }
    if (!want_logits) return;
    double tl = now(); static float hn[D]; rmsnorm(h, FNORM, hn, D);
    if (FAST >= 4 && !F16EMB) { head_cert(hn, logits); T_LOG += now() - tl; return; }
    #pragma omp parallel for schedule(static)
    for (int vi = 0; vi < V; vi++) {
        const uint16_t *w = EMB + (size_t)vi * D; __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < D; i += 8) {
            __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const void *)(w + i))), 16));
            if (F16EMB) wf = _mm256_cvtph_ps(_mm256_cvtps_ph(wf, _MM_FROUND_TO_NEAREST_INT));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wf, _mm256_loadu_ps(hn + i)));
        }
        float a[8]; _mm256_storeu_ps(a, acc);
        logits[vi] = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
    }
    T_LOG += now() - tl;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s MODEL NGEN THREADS tok...\n", argv[0]); return 1; }
    FAST = getenv("ABN_FAST") ? atoi(getenv("ABN_FAST")) : 0;
    int ngen = atoi(argv[2]); omp_set_num_threads(atoi(argv[3])); SCALE_DIV = getenv("ABN_SCALE_DIV") != NULL; ROPE_IL = getenv("ABN_ROPE_IL") != NULL; ACT_SILU = getenv("ABN_SILU") != NULL; F16KV = getenv("ABN_F16KV") != NULL; F16EMB = getenv("ABN_F16EMB") != NULL; SCALE_MODE = getenv("ABN_NOSCALE") != NULL ? 2 : 0;
    int np = argc - 4, toks[MAXCTX]; for (int i = 0; i < np; i++) toks[i] = atoi(argv[4 + i]);
    crt_init();
    if (getenv("ABN_LEDGER")) { LEDGER = 1; LF = fopen(getenv("ABN_LEDGER"), "w"); double a = now(); uint64_t c0 = __rdtsc(); while (now() - a < 0.1) ; TSCHZ = (double)(__rdtsc() - c0) / (now() - a); }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    load_model(argv[1]);
    if (FAST >= 4 && !F16EMB) head_init();
    clock_gettime(CLOCK_MONOTONIC, &t1); fprintf(stderr, "load %.2fs\n", (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
    LUT = aligned_alloc(64, (size_t)FF / 2 * R * 32); RES = aligned_alloc(64, (size_t)R * FF); XQ = aligned_alloc(64, FF);
    for (int i = 0; i < NL; i++) { KC[i] = calloc((size_t)MAXCTX * NKV * HD, 4); VC[i] = calloc((size_t)MAXCTX * NKV * HD, 4); }
    double lnth = 13.122363377404328;       /* ln(500000) */
    for (int i = 0; i < HD / 2; i++) INVF[i] = dexp(-lnth * (double)i / 64.0);
    float *h = malloc(D * 4), *logits = malloc((size_t)V * 4);
    FILE *lf = getenv("ABN_LOGITS") ? fopen(getenv("ABN_LOGITS"), "wb") : NULL;
    if (getenv("ABN_PPL")) {                /* OUTREACH-P1-FOLLOWUP: llama.cpp-perplexity protocol (n_ctx 512, BOS first in every chunk, score the second half) */
        FILE *pf = fopen(getenv("ABN_PPL"), "r"); int ids[1 << 16], nid = 0; if (!pf) return 6;
        while (nid < (1 << 16) && fscanf(pf, "%d", &ids[nid]) == 1) nid++; fclose(pf);
        double nll = 0; long cnt = 0; int nchunk = nid / MAXCTX;
        for (int ch = 0; ch < nchunk; ch++) {
            int *ct = ids + (size_t)ch * MAXCTX; int save = ct[0]; ct[0] = 128000;   /* BOS */
            double cn = 0; long cc = 0;
            for (int j = 0; j < MAXCTX - 1; j++) {
                int sc2 = j >= MAXCTX / 2; forward(ct[j], j, h, logits, sc2);
                if (!sc2) continue;
                double mx = logits[0]; for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
                double z = 0; for (int i = 0; i < V; i++) z += exp((double)logits[i] - mx);
                cn += -((double)logits[ct[j + 1]] - mx - log(z)); cc++;
            }
            ct[0] = save; nll += cn; cnt += cc;
            fprintf(stderr, "ppl chunk %d: %.4f (running %.4f)\n", ch, exp(cn / cc), exp(nll / cnt));
        }
        printf("PPL %.4f over %ld tokens in %d chunks (act=%s, f16kv=%d)\n", exp(nll / cnt), cnt, nchunk, ACT_SILU ? "silu" : "relu2", F16KV);
        return 0;
    }
    int pos = 0; double tgen = 0; int ngenerated = 0;
    for (; pos < np - 1; pos++) forward(toks[pos], pos, h, logits, 0);   /* prefill */
    int cur = toks[np - 1];
    for (int g = 0; g < ngen && pos < MAXCTX; g++, pos++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (LEDGER) { memset(LBUSY, 0, sizeof LBUSY); memset(LWALL, 0, sizeof LWALL); memset(LBYTES, 0, sizeof LBYTES); }
        forward(cur, pos, h, logits, 1);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (LEDGER) ledger_emit(g, (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec), atoi(argv[3])); tgen += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        int best = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
        if (lf) fwrite(logits, 4, V, lf);
        printf("%d ", best); fflush(stdout); cur = best; ngenerated++;
    }
    printf("\n"); if (lf) fclose(lf);
    fprintf(stderr, "decode %d tokens in %.3fs = %.3f tok/s (%d threads, prefill %d tokens excluded)\n", ngenerated, tgen, ngenerated / tgen, atoi(argv[3]), np - 1);
    if (N_STEPS) fprintf(stderr, "head survivors: mean %.1f max %ld of %d over %ld steps\n", (double)N_SURV / N_STEPS, MAX_SURV, V, N_STEPS);
    fprintf(stderr, "time split (all tokens incl prefill): precompute %.2fs pull %.2fs crt %.2fs logits %.2fs forward-total %.2fs\n", T_PRE, T_PULL, T_CRT, T_LOG, T_FWD);
    return 0;
}
