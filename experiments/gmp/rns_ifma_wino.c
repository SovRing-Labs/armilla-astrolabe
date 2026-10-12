/* Armilla vs GMP, step 4 (2026-10-12): step 3 + WINOGRAD'S INNER PRODUCT (1968) in both base extensions.
 * sum_i xi_i T_ij = sum_k (xi_2k + T_2k+1,j)(xi_2k+1 + T_2k,j) - sum_k xi_2k xi_2k+1 - sum_k T_2k,j T_2k+1,j :
 * the table half is a constant per output (precomputed), the message half is one scalar per message, so the extension needs
 * n/2 instead of n products per output (inputs < 2^49 still fit IFMA's 52 bits).
 * --- step 3 header follows ---
 * Armilla vs GMP, step 3 (2026-10-12): batched, windowed, constant-time, multi-core RNS Montgomery modexp with AVX-512 IFMA.
 *  - rings: primes 2^48 - c (see rns_ifma.c for the reduction);
 *  - BATCH of 8 messages per call: in the base extension each table vector T[i][j..j+7] is loaded once and used by all 8 messages
 *    (16 IFMA accumulators: lo/hi per message), the table is read 8x less per message;
 *  - fixed 5-bit windows: x^0..x^31 precomputed, then per window 5 squarings + 1 multiply (about 1.2 multiplies per bit, against
 *    1.5 for plain square-and-multiply);
 *  - CONSTANT TIME with respect to the exponent: every window performs the same operations; the table entry is selected by
 *    scanning all 32 entries with masks (no secret-dependent branch or address);
 *  - multi-core: OpenMP over batches.
 */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define BB 8
#define NMAX 384
#define M52 0xFFFFFFFFFFFFFull
#define M48 0xFFFFFFFFFFFFull

typedef struct {
    int n;
    const uint64_t *a, *b, *ca, *cb; const double *ia, *ib;
    const uint64_t *cA, *TAB, *MAb, *Nb, *dB, *TBA, *MBa;
    const uint64_t *CTAB, *CTBA;            /* sum_k T[2k][j] T[2k+1][j] mod (b_j | a_i) */
} Ctx;

static inline __m512i red(__m512i lo, __m512i hi, __m512i m, __m512i c, __m512i c16) {
    __m512i H = _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52));
    __m512i L = _mm512_and_si512(lo, _mm512_set1_epi64(M52));
    __m512i Plo = _mm512_madd52lo_epu64(L, H, c16);
    __m512i Phi = _mm512_madd52hi_epu64(_mm512_setzero_si512(), H, c16);
    __m512i T = _mm512_add_epi64(_mm512_mullo_epi64(Phi, c16), Plo);
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

static inline __m512i big_mod(unsigned __int128 C, __m512i m, __m512i c, __m512i c16) {
    uint64_t c0 = (uint64_t)C & M48, c1 = (uint64_t)(C >> 48) & M48, c2 = (uint64_t)(C >> 96);
    /* 2^48 == c and 2^96 == c^2 (mod m) */
    __m512i v = _mm512_add_epi64(mulmod(_mm512_set1_epi64(c1), c, m, c, c16), _mm512_set1_epi64(c0));
    v = _mm512_add_epi64(v, mulmod(mulmod(_mm512_set1_epi64(c2), c, m, c, c16), c, m, c, c16));
    for (int k = 0; k < 3; k++) v = _mm512_mask_sub_epi64(v, _mm512_cmpge_epu64_mask(v, m), v, m);
    return v;
}
static inline unsigned __int128 pair_sum(const uint64_t *x, int n) {
    unsigned __int128 C = 0;
    for (int k = 0; k < n; k += 2) C += (unsigned __int128)x[k] * x[k + 1];
    return C;
}

/* one value = n residues in base A followed by n residues in base B (2n uint64) */
/* R[b] = X[b] * Y[b] * A^-1 mod N for b < BB; all arrays [BB][2n] */
static void mont8(const Ctx *X, const uint64_t *Xv, const uint64_t *Yv, uint64_t *Rv) {
    const int n = X->n, S = 2 * n;
    uint64_t xi[BB][NMAX] __attribute__((aligned(64))), qb[BB][NMAX] __attribute__((aligned(64))), rr[BB][NMAX] __attribute__((aligned(64)));
    uint64_t alpha[BB], beta[BB];
    for (int q = 0; q < BB; q++) {
        double fs = 0;
        for (int i = 0; i < n; i += 8) {
            LOADM(X->a, X->ca, i);
            __m512i v = mulmod(mulmod(_mm512_loadu_si512(Xv + q * S + i), _mm512_loadu_si512(Yv + q * S + i), m, c, c16),
                               _mm512_loadu_si512(X->cA + i), m, c, c16);
            _mm512_store_si512(xi[q] + i, v);
            fs += _mm512_reduce_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(v), _mm512_loadu_pd(X->ia + i)));
        }
        alpha[q] = (uint64_t)floor(fs);
    }
    unsigned __int128 CX[BB];
    for (int q = 0; q < BB; q++) CX[q] = pair_sum(xi[q], n);
    for (int j = 0; j < n; j += 8) {                                   /* A -> B, one table load per 8 messages */
        __m512i lo[BB], hi[BB], part[BB];
        LOADM(X->b, X->cb, j);
        for (int q = 0; q < BB; q++) part[q] = _mm512_setzero_si512();
        for (int i0 = 0; i0 < n; i0 += 120) {
        int i1 = i0 + 120 < n ? i0 + 120 : n;
        for (int q = 0; q < BB; q++) { lo[q] = _mm512_setzero_si512(); hi[q] = _mm512_setzero_si512(); }
        for (int i = i0; i < i1; i += 2) {
            __m512i t0 = _mm512_loadu_si512(X->TAB + (size_t)i * n + j), t1 = _mm512_loadu_si512(X->TAB + (size_t)(i + 1) * n + j);
            for (int q = 0; q < BB; q++) {
                __m512i u = _mm512_add_epi64(_mm512_set1_epi64(xi[q][i]), t1), w = _mm512_add_epi64(_mm512_set1_epi64(xi[q][i + 1]), t0);
                lo[q] = _mm512_madd52lo_epu64(lo[q], u, w); hi[q] = _mm512_madd52hi_epu64(hi[q], u, w);
            }
        }
        for (int q = 0; q < BB; q++) {
            part[q] = _mm512_add_epi64(part[q], red(lo[q], hi[q], m, c, c16));
            part[q] = _mm512_mask_sub_epi64(part[q], _mm512_cmpge_epu64_mask(part[q], m), part[q], m);
        }
        }
        __m512i mab = _mm512_loadu_si512(X->MAb + j), ct = _mm512_loadu_si512(X->CTAB + j);
        for (int q = 0; q < BB; q++) {
            __m512i w = part[q];
            __m512i corr = _mm512_add_epi64(_mm512_add_epi64(big_mod(CX[q], m, c, c16), ct), mulmod(_mm512_set1_epi64(alpha[q]), mab, m, c, c16));
            for (int k = 0; k < 2; k++) corr = _mm512_mask_sub_epi64(corr, _mm512_cmpge_epu64_mask(corr, m), corr, m);
            __m512i v = _mm512_sub_epi64(w, corr);
            v = _mm512_mask_add_epi64(v, _mm512_cmplt_epi64_mask(v, _mm512_setzero_si512()), v, m);
            _mm512_store_si512(qb[q] + j, v);
        }
    }
    for (int q = 0; q < BB; q++) {
        double fs = 0;
        for (int j = 0; j < n; j += 8) {
            LOADM(X->b, X->cb, j);
            __m512i t = _mm512_add_epi64(mulmod(_mm512_loadu_si512(Xv + q * S + n + j), _mm512_loadu_si512(Yv + q * S + n + j), m, c, c16),
                                         mulmod(_mm512_load_si512(qb[q] + j), _mm512_loadu_si512(X->Nb + j), m, c, c16));
            t = _mm512_mask_sub_epi64(t, _mm512_cmpge_epu64_mask(t, m), t, m);
            __m512i v = mulmod(t, _mm512_loadu_si512(X->dB + j), m, c, c16);
            _mm512_store_si512(rr[q] + j, v);
            fs += _mm512_reduce_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(v), _mm512_loadu_pd(X->ib + j)));
        }
        beta[q] = (uint64_t)llrint(fs);
    }
    unsigned __int128 CR[BB];
    for (int q = 0; q < BB; q++) CR[q] = pair_sum(rr[q], n);
    for (int i = 0; i < n; i += 8) {                                   /* B -> A */
        __m512i lo[BB], hi[BB], part[BB];
        LOADM(X->a, X->ca, i);
        for (int q = 0; q < BB; q++) part[q] = _mm512_setzero_si512();
        for (int j0 = 0; j0 < n; j0 += 120) {
        int j1 = j0 + 120 < n ? j0 + 120 : n;
        for (int q = 0; q < BB; q++) { lo[q] = _mm512_setzero_si512(); hi[q] = _mm512_setzero_si512(); }
        for (int j = j0; j < j1; j += 2) {
            __m512i t0 = _mm512_loadu_si512(X->TBA + (size_t)j * n + i), t1 = _mm512_loadu_si512(X->TBA + (size_t)(j + 1) * n + i);
            for (int q = 0; q < BB; q++) {
                __m512i u = _mm512_add_epi64(_mm512_set1_epi64(rr[q][j]), t1), w = _mm512_add_epi64(_mm512_set1_epi64(rr[q][j + 1]), t0);
                lo[q] = _mm512_madd52lo_epu64(lo[q], u, w); hi[q] = _mm512_madd52hi_epu64(hi[q], u, w);
            }
        }
        for (int q = 0; q < BB; q++) {
            part[q] = _mm512_add_epi64(part[q], red(lo[q], hi[q], m, c, c16));
            part[q] = _mm512_mask_sub_epi64(part[q], _mm512_cmpge_epu64_mask(part[q], m), part[q], m);
        }
        }
        __m512i mba = _mm512_loadu_si512(X->MBa + i), ct = _mm512_loadu_si512(X->CTBA + i);
        for (int q = 0; q < BB; q++) {
            __m512i w = part[q];
            __m512i corr = _mm512_add_epi64(_mm512_add_epi64(big_mod(CR[q], m, c, c16), ct), mulmod(_mm512_set1_epi64(beta[q]), mba, m, c, c16));
            for (int k = 0; k < 2; k++) corr = _mm512_mask_sub_epi64(corr, _mm512_cmpge_epu64_mask(corr, m), corr, m);
            __m512i v = _mm512_sub_epi64(w, corr);
            v = _mm512_mask_add_epi64(v, _mm512_cmplt_epi64_mask(v, _mm512_setzero_si512()), v, m);
            _mm512_storeu_si512(Rv + q * S + i, v);
        }
    }
    for (int q = 0; q < BB; q++)
        for (int j = 0; j < n; j += 8) {
            LOADM(X->b, X->cb, j);
            _mm512_storeu_si512(Rv + q * S + n + j, mulmod(_mm512_load_si512(rr[q] + j), _mm512_loadu_si512(X->dB + n + j), m, c, c16));
        }
}

/* constant-time select: dst[b] = table[w[b]] for every message, scanning all 32 entries */
static void ct_select(const uint64_t *tab /*[32][BB][2n]*/, const uint8_t *w /*[BB]*/, int S, uint64_t *dst /*[BB][2n]*/) {
    for (int q = 0; q < BB; q++) {
        for (int k = 0; k < S; k += 8) {
            __m512i acc = _mm512_setzero_si512();
            for (int e = 0; e < 32; e++) {
                __mmask8 hit = (__mmask8)(0 - (uint8_t)(w[q] == e));   /* compiled to setcc/neg: no branch */
                acc = _mm512_mask_mov_epi64(acc, hit, _mm512_loadu_si512(tab + ((size_t)e * BB + q) * S + k));
            }
            _mm512_storeu_si512(dst + q * S + k, acc);
        }
    }
}

#define ARGS int n, const uint64_t *a, const uint64_t *b, const uint64_t *ca, const uint64_t *cb, const double *ia, const double *ib, \
    const uint64_t *cA, const uint64_t *TAB, const uint64_t *MAb, const uint64_t *Nb, const uint64_t *dB, const uint64_t *TBA, const uint64_t *MBa, \
    const uint64_t *CTAB, const uint64_t *CTBA
#define CTX Ctx X = {n, a, b, ca, cb, ia, ib, cA, TAB, MAb, Nb, dB, TBA, MBa, CTAB, CTBA}

/* nb batches of 8 messages. xs, ones: [nb*8][2n] (Montgomery forms). Each message has its OWN exponent:
 * windows: [nb*8][nwin] (5-bit digits, most significant first). out: [nb*8][2n]. Threads over batches. */
void batch_modexp_ct(ARGS, int nb, const uint64_t *xs, const uint64_t *ones, const uint8_t *windows, int nwin, uint64_t *out, int threads) {
    CTX; const int S = 2 * n;
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int bt = 0; bt < nb; bt++) {
        uint64_t *tab = aligned_alloc(64, (size_t)32 * BB * S * 8);
        uint64_t *acc = aligned_alloc(64, (size_t)BB * S * 8), *tmp = aligned_alloc(64, (size_t)BB * S * 8), *sel = aligned_alloc(64, (size_t)BB * S * 8);
        const uint64_t *x = xs + (size_t)bt * BB * S;
        memcpy(tab, ones + (size_t)bt * BB * S, (size_t)BB * S * 8);                 /* x^0 = one (Montgomery) */
        memcpy(tab + (size_t)BB * S, x, (size_t)BB * S * 8);                          /* x^1 */
        for (int e = 2; e < 32; e++) mont8(&X, tab + (size_t)(e - 1) * BB * S, x, tab + (size_t)e * BB * S);
        uint8_t w[BB];
        for (int q = 0; q < BB; q++) w[q] = windows[((size_t)bt * BB + q) * nwin];
        ct_select(tab, w, S, acc);
        for (int k = 1; k < nwin; k++) {
            for (int s = 0; s < 5; s++) { mont8(&X, acc, acc, tmp); memcpy(acc, tmp, (size_t)BB * S * 8); }
            for (int q = 0; q < BB; q++) w[q] = windows[((size_t)bt * BB + q) * nwin + k];
            ct_select(tab, w, S, sel);
            mont8(&X, acc, sel, tmp); memcpy(acc, tmp, (size_t)BB * S * 8);
        }
        memcpy(out + (size_t)bt * BB * S, acc, (size_t)BB * S * 8);
        free(tab); free(acc); free(tmp); free(sel);
    }
}

/* convert out of the Montgomery domain (multiply by 1), one batch */
void batch_mont(ARGS, const uint64_t *xs, const uint64_t *ys, uint64_t *out) { CTX; mont8(&X, xs, ys, out); }
