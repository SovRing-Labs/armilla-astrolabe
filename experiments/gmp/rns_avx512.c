/* Armilla — RNS Montgomery modular exponentiation in C / AVX-512 (2026-10-12).
 * Two bases of n 28-bit primes each (A, B), every residue in its own 64-bit lane; no carries anywhere, so all work is SIMD.
 *  mulmod: vpmuludq (28 x 28 -> 56 bits), quotient from a double reciprocal, one correction each way.
 *  base extension: s_j = sum_i xi_i * T[i][j] accumulated WITHOUT reduction in uint64 lanes (n <= 255 terms of < 2^56),
 *  reduced once per output; the correction count alpha = floor / rint of sum xi_i / m_i (Kawamura et al. 2000).
 *  Montgomery in RNS: Posch & Posch 1995; Bajard, Didier & Kornerup 1998.
 */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n;                                   /* moduli per base, multiple of 8 */
    const uint64_t *a, *b;                   /* moduli */
    const double *ia, *ib;                   /* 1 / m */
    const uint64_t *cA;                      /* (-N^-1 * A_i^-1) mod a_i */
    const uint32_t *TAB;                     /* [i][j] = A_i mod b_j   (n x n, 32-bit) */
    const uint64_t *MAb;                     /* A mod b_j */
    const uint64_t *Nb;                      /* N mod b_j */
    const uint64_t *dB;                      /* (A^-1 * B_j^-1) mod b_j */
    const uint32_t *TBA;                     /* [j][i] = B_j mod a_i   (n x n, 32-bit) */
    const uint64_t *MBa;                     /* B mod a_i */
} Ctx;

static inline __m512i mulmod(__m512i x, __m512i y, __m512i m, __m512d im) {
    __m512i p = _mm512_mul_epu32(x, y);                                  /* < 2^56 */
    __m512i q = _mm512_cvttpd_epu64(_mm512_mul_pd(_mm512_cvtepu64_pd(p), im));
    __m512i r = _mm512_sub_epi64(p, _mm512_mul_epu32(q, m));             /* q < 2^29 fits 32 bits */
    r = _mm512_mask_add_epi64(r, _mm512_cmplt_epi64_mask(r, _mm512_setzero_si512()), r, m);
    r = _mm512_mask_sub_epi64(r, _mm512_cmpge_epi64_mask(r, m), r, m);
    return r;
}
static inline __m512i redu64(__m512i s, __m512i m, __m512d im) {          /* s < 2^64 -> s mod m */
    __m512i q = _mm512_cvttpd_epu64(_mm512_mul_pd(_mm512_cvtepu64_pd(s), im));
    __m512i r = _mm512_sub_epi64(s, _mm512_mullo_epi64(q, m));           /* exact mod 2^64; true r in (-2m, 2m) */
    r = _mm512_mask_add_epi64(r, _mm512_cmplt_epi64_mask(r, _mm512_setzero_si512()), r, m);
    r = _mm512_mask_add_epi64(r, _mm512_cmplt_epi64_mask(r, _mm512_setzero_si512()), r, m);
    r = _mm512_mask_sub_epi64(r, _mm512_cmpge_epi64_mask(r, m), r, m);
    r = _mm512_mask_sub_epi64(r, _mm512_cmpge_epi64_mask(r, m), r, m);
    return r;
}

/* out = x * y * A^-1 mod N (< 3N), x and y given in both bases */
/* s[j..j+31] = sum_i xi[i] * T[i][j..j+31]: four 8-lane accumulators per pass, table rows read as uint32 */
static inline void ext4(const uint64_t *xi, const uint32_t *T, int n, int j, __m512i *s) {
    __m512i s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0;
    for (int i = 0; i < n; i++) {
        __m512i x = _mm512_set1_epi64(xi[i]);
        const uint32_t *t = T + (size_t)i * n + j;
        s0 = _mm512_add_epi64(s0, _mm512_mul_epu32(x, _mm512_cvtepu32_epi64(_mm256_loadu_si256((const void *)(t)))));
        s1 = _mm512_add_epi64(s1, _mm512_mul_epu32(x, _mm512_cvtepu32_epi64(_mm256_loadu_si256((const void *)(t + 8)))));
        s2 = _mm512_add_epi64(s2, _mm512_mul_epu32(x, _mm512_cvtepu32_epi64(_mm256_loadu_si256((const void *)(t + 16)))));
        s3 = _mm512_add_epi64(s3, _mm512_mul_epu32(x, _mm512_cvtepu32_epi64(_mm256_loadu_si256((const void *)(t + 24)))));
    }
    s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
}

static void mont(const Ctx *c, const uint64_t *xa, const uint64_t *xb, const uint64_t *ya, const uint64_t *yb,
                 uint64_t *ra, uint64_t *rb) {
    int n = c->n;
    uint64_t xi[256] __attribute__((aligned(64))), qb[256] __attribute__((aligned(64)));
    double fs = 0;
    for (int i = 0; i < n; i += 8) {
        __m512i m = _mm512_loadu_si512(c->a + i); __m512d im = _mm512_loadu_pd(c->ia + i);
        __m512i v = mulmod(mulmod(_mm512_loadu_si512(xa + i), _mm512_loadu_si512(ya + i), m, im), _mm512_loadu_si512(c->cA + i), m, im);
        _mm512_store_si512(xi + i, v);
        fs += _mm512_reduce_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(v), im));
    }
    uint64_t alpha = (uint64_t)floor(fs);
    for (int j0 = 0; j0 < n; j0 += 32) {                                   /* extension A -> B */
      __m512i S4[4]; ext4(xi, c->TAB, n, j0, S4);
      for (int u = 0; u < 4; u++) { int j = j0 + 8 * u; __m512i s = S4[u];
        __m512i m = _mm512_loadu_si512(c->b + j); __m512d im = _mm512_loadu_pd(c->ib + j);
        __m512i v = redu64(s, m, im);
        __m512i corr = mulmod(_mm512_set1_epi64(alpha), _mm512_loadu_si512(c->MAb + j), m, im);
        v = _mm512_sub_epi64(v, corr);
        v = _mm512_mask_add_epi64(v, _mm512_cmplt_epi64_mask(v, _mm512_setzero_si512()), v, m);
        _mm512_store_si512(qb + j, v);
    } }
    fs = 0;
    for (int j = 0; j < n; j += 8) {                                       /* r = (xy + qN) A^-1, then xi' = r B_j^-1 */
        __m512i m = _mm512_loadu_si512(c->b + j); __m512d im = _mm512_loadu_pd(c->ib + j);
        __m512i t = _mm512_add_epi64(mulmod(_mm512_loadu_si512(xb + j), _mm512_loadu_si512(yb + j), m, im),
                                     mulmod(_mm512_load_si512(qb + j), _mm512_loadu_si512(c->Nb + j), m, im));
        t = _mm512_mask_sub_epi64(t, _mm512_cmpge_epi64_mask(t, m), t, m);
        _mm512_storeu_si512(rb + j, mulmod(t, _mm512_loadu_si512(c->dB + j), m, im));   /* temporarily xi' */
        fs += _mm512_reduce_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(_mm512_loadu_si512(rb + j)), im));
    }
    uint64_t beta = (uint64_t)llrint(fs);                                   /* exact: r / B is tiny */
    for (int i0 = 0; i0 < n; i0 += 32) {                                   /* extension B -> A */
      __m512i S4[4]; ext4(rb, c->TBA, n, i0, S4);
      for (int u = 0; u < 4; u++) { int i = i0 + 8 * u; __m512i s = S4[u];
        __m512i m = _mm512_loadu_si512(c->a + i); __m512d im = _mm512_loadu_pd(c->ia + i);
        __m512i v = redu64(s, m, im);
        __m512i corr = mulmod(_mm512_set1_epi64(beta), _mm512_loadu_si512(c->MBa + i), m, im);
        v = _mm512_sub_epi64(v, corr);
        v = _mm512_mask_add_epi64(v, _mm512_cmplt_epi64_mask(v, _mm512_setzero_si512()), v, m);
        _mm512_storeu_si512(ra + i, v);
    } }
    /* rb currently holds xi' = r * B_j^-1; turn it back into r_j = xi'_j * B_j mod b_j */
    for (int j = 0; j < n; j += 8) {
        __m512i m = _mm512_loadu_si512(c->b + j); __m512d im = _mm512_loadu_pd(c->ib + j);
        /* dB holds 2n entries: [0, n) = (A^-1 * B_j^-1) mod b_j, [n, 2n) = B_j mod b_j */
        _mm512_storeu_si512(rb + j, mulmod(_mm512_loadu_si512(rb + j), _mm512_loadu_si512(c->dB + n + j), m, im));
    }
}

int rns_ctx_size(void) { return sizeof(Ctx); }

/* modexp: base in both bases (already in Montgomery form), one in Montgomery form, exponent bits MSB first */
void rns_modexp(int n, const uint64_t *a, const uint64_t *b, const double *ia, const double *ib, const uint64_t *cA,
                const uint32_t *TAB, const uint64_t *MAb, const uint64_t *Nb, const uint64_t *dB, const uint32_t *TBA,
                const uint64_t *MBa, const uint64_t *xa, const uint64_t *xb, const uint64_t *onea, const uint64_t *oneb,
                const uint8_t *ebits, int nbits, uint64_t *outa, uint64_t *outb, int *nmuls) {
    Ctx c = {n, a, b, ia, ib, cA, TAB, MAb, Nb, dB, TBA, MBa};
    uint64_t acc_a[256], acc_b[256], t_a[256], t_b[256];
    memcpy(acc_a, onea, n * 8); memcpy(acc_b, oneb, n * 8);
    int cnt = 0;
    for (int k = 0; k < nbits; k++) {
        mont(&c, acc_a, acc_b, acc_a, acc_b, t_a, t_b); cnt++;
        if (ebits[k]) { mont(&c, t_a, t_b, xa, xb, acc_a, acc_b); cnt++; }
        else { memcpy(acc_a, t_a, n * 8); memcpy(acc_b, t_b, n * 8); }
    }
    memcpy(outa, acc_a, n * 8); memcpy(outb, acc_b, n * 8); *nmuls = cnt;
}

void rns_mont(int n, const uint64_t *a, const uint64_t *b, const double *ia, const double *ib, const uint64_t *cA,
              const uint32_t *TAB, const uint64_t *MAb, const uint64_t *Nb, const uint64_t *dB, const uint32_t *TBA,
              const uint64_t *MBa, const uint64_t *xa, const uint64_t *xb, const uint64_t *ya, const uint64_t *yb,
              uint64_t *ra, uint64_t *rb) {
    Ctx c = {n, a, b, ia, ib, cA, TAB, MAb, Nb, dB, TBA, MBa};
    mont(&c, xa, xb, ya, yb, ra, rb);
}
