/* Armilla vs GMP — RNS Montgomery with AVX-512 IFMA (2026-10-12).
 * Rings: primes m = 2^48 - c (c < 2^16), one residue per 64-bit lane. IFMA (vpmadd52lo/hi) multiplies 52-bit inputs into a
 * 104-bit product split at bit 52, 8 lanes per instruction; the base extension adds n such products per lane WITHOUT reducing
 * (lo < n*2^52, hi < n*2^44) and reduces once:
 *   S = H*2^52 + L  ==  H*(16c) + L  (mod m)      because 2^52 = 16 * 2^48 == 16c
 *     -> again split by IFMA -> T < 2^54 -> T = Th*2^48 + Tl == Th*c + Tl -> one or two conditional subtractions.
 * Montgomery in RNS: Posch & Posch 1995; Bajard, Didier & Kornerup 1998; correction count alpha: Kawamura et al. 2000.
 * Pseudo-Mersenne moduli for RNS: field-known (Bajard et al.).
 */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n;
    const uint64_t *a, *b, *ca, *cb;          /* moduli and their c = 2^48 - m */
    const double *ia, *ib;
    const uint64_t *cA, *TAB, *MAb, *Nb, *dB, *TBA, *MBa;   /* dB: [0,n) A^-1 B_j^-1 mod b_j, [n,2n) B_j mod b_j */
} Ctx;

#define M52 0xFFFFFFFFFFFFFull
#define M48 0xFFFFFFFFFFFFull

static inline __m512i red(__m512i lo, __m512i hi, __m512i m, __m512i c, __m512i c16) {
    __m512i H = _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52));
    __m512i L = _mm512_and_si512(lo, _mm512_set1_epi64(M52));
    __m512i Plo = _mm512_madd52lo_epu64(L, H, c16);
    __m512i Phi = _mm512_madd52hi_epu64(_mm512_setzero_si512(), H, c16);
    __m512i T = _mm512_add_epi64(_mm512_mullo_epi64(Phi, c16), Plo);                 /* < 2^54 */
    T = _mm512_add_epi64(_mm512_mullo_epi64(_mm512_srli_epi64(T, 48), c), _mm512_and_si512(T, _mm512_set1_epi64(M48)));
    T = _mm512_mask_sub_epi64(T, _mm512_cmpge_epu64_mask(T, m), T, m);
    T = _mm512_mask_sub_epi64(T, _mm512_cmpge_epu64_mask(T, m), T, m);
    return T;
}
static inline __m512i mulmod(__m512i x, __m512i y, __m512i m, __m512i c, __m512i c16) {
    __m512i z = _mm512_setzero_si512();
    return red(_mm512_madd52lo_epu64(z, x, y), _mm512_madd52hi_epu64(z, x, y), m, c, c16);
}
#define LOADM(arr, cc, k) __m512i m = _mm512_loadu_si512((arr) + (k)), c = _mm512_loadu_si512((cc) + (k)), c16 = _mm512_slli_epi64(c, 4)

static void mont(const Ctx *X, const uint64_t *xa, const uint64_t *xb, const uint64_t *ya, const uint64_t *yb, uint64_t *ra, uint64_t *rb) {
    int n = X->n;
    uint64_t xi[256] __attribute__((aligned(64))), qb[256] __attribute__((aligned(64)));
    double fs = 0;
    for (int i = 0; i < n; i += 8) {
        LOADM(X->a, X->ca, i);
        __m512i v = mulmod(mulmod(_mm512_loadu_si512(xa + i), _mm512_loadu_si512(ya + i), m, c, c16), _mm512_loadu_si512(X->cA + i), m, c, c16);
        _mm512_store_si512(xi + i, v);
        fs += _mm512_reduce_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(v), _mm512_loadu_pd(X->ia + i)));
    }
    uint64_t alpha = (uint64_t)floor(fs);
    for (int j = 0; j < n; j += 8) {                                   /* extension A -> B, lazy IFMA sums */
        __m512i lo = _mm512_setzero_si512(), hi = _mm512_setzero_si512();
        for (int i = 0; i < n; i++) {
            __m512i x = _mm512_set1_epi64(xi[i]), t = _mm512_loadu_si512(X->TAB + (size_t)i * n + j);
            lo = _mm512_madd52lo_epu64(lo, x, t); hi = _mm512_madd52hi_epu64(hi, x, t);
        }
        LOADM(X->b, X->cb, j);
        __m512i v = red(lo, hi, m, c, c16);
        __m512i corr = mulmod(_mm512_set1_epi64(alpha), _mm512_loadu_si512(X->MAb + j), m, c, c16);
        v = _mm512_sub_epi64(v, corr);
        v = _mm512_mask_add_epi64(v, _mm512_cmplt_epi64_mask(v, _mm512_setzero_si512()), v, m);
        _mm512_store_si512(qb + j, v);
    }
    fs = 0;
    for (int j = 0; j < n; j += 8) {
        LOADM(X->b, X->cb, j);
        __m512i t = _mm512_add_epi64(mulmod(_mm512_loadu_si512(xb + j), _mm512_loadu_si512(yb + j), m, c, c16),
                                     mulmod(_mm512_load_si512(qb + j), _mm512_loadu_si512(X->Nb + j), m, c, c16));
        t = _mm512_mask_sub_epi64(t, _mm512_cmpge_epu64_mask(t, m), t, m);
        __m512i v = mulmod(t, _mm512_loadu_si512(X->dB + j), m, c, c16);
        _mm512_storeu_si512(rb + j, v);
        fs += _mm512_reduce_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(v), _mm512_loadu_pd(X->ib + j)));
    }
    uint64_t beta = (uint64_t)llrint(fs);
    for (int i = 0; i < n; i += 8) {                                   /* extension B -> A */
        __m512i lo = _mm512_setzero_si512(), hi = _mm512_setzero_si512();
        for (int j = 0; j < n; j++) {
            __m512i x = _mm512_set1_epi64(rb[j]), t = _mm512_loadu_si512(X->TBA + (size_t)j * n + i);
            lo = _mm512_madd52lo_epu64(lo, x, t); hi = _mm512_madd52hi_epu64(hi, x, t);
        }
        LOADM(X->a, X->ca, i);
        __m512i v = red(lo, hi, m, c, c16);
        __m512i corr = mulmod(_mm512_set1_epi64(beta), _mm512_loadu_si512(X->MBa + i), m, c, c16);
        v = _mm512_sub_epi64(v, corr);
        v = _mm512_mask_add_epi64(v, _mm512_cmplt_epi64_mask(v, _mm512_setzero_si512()), v, m);
        _mm512_storeu_si512(ra + i, v);
    }
    for (int j = 0; j < n; j += 8) {
        LOADM(X->b, X->cb, j);
        _mm512_storeu_si512(rb + j, mulmod(_mm512_loadu_si512(rb + j), _mm512_loadu_si512(X->dB + n + j), m, c, c16));
    }
}

#define ARGS int n, const uint64_t *a, const uint64_t *b, const uint64_t *ca, const uint64_t *cb, const double *ia, const double *ib, \
    const uint64_t *cA, const uint64_t *TAB, const uint64_t *MAb, const uint64_t *Nb, const uint64_t *dB, const uint64_t *TBA, const uint64_t *MBa
#define CTX Ctx X = {n, a, b, ca, cb, ia, ib, cA, TAB, MAb, Nb, dB, TBA, MBa}

void ifma_modexp(ARGS, const uint64_t *xa, const uint64_t *xb, const uint64_t *onea, const uint64_t *oneb,
                 const uint8_t *ebits, int nbits, uint64_t *outa, uint64_t *outb, int *nmuls) {
    CTX; uint64_t A0[256], B0[256], A1[256], B1[256]; int cnt = 0;
    memcpy(A0, onea, n * 8); memcpy(B0, oneb, n * 8);
    for (int k = 0; k < nbits; k++) {
        mont(&X, A0, B0, A0, B0, A1, B1); cnt++;
        if (ebits[k]) { mont(&X, A1, B1, xa, xb, A0, B0); cnt++; }
        else { memcpy(A0, A1, n * 8); memcpy(B0, B1, n * 8); }
    }
    memcpy(outa, A0, n * 8); memcpy(outb, B0, n * 8); *nmuls = cnt;
}
void ifma_mont(ARGS, const uint64_t *xa, const uint64_t *xb, const uint64_t *ya, const uint64_t *yb, uint64_t *ra, uint64_t *rb) {
    CTX; mont(&X, xa, xb, ya, yb, ra, rb);
}
