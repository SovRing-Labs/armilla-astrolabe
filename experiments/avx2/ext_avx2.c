/* AVX2-only base extension (no AVX-512, no IFMA): 28-bit rings, 4 messages per pass, 64-bit lanes (2026-10-12).
 * plain:    s_j = sum_i xi_i T_ij                       (vpmuludq + vpaddq per product)
 * winograd: s_j = sum_k (xi_2k + T_2k+1,j)(xi_2k+1 + T_2k,j) - sum_k xi_2k xi_2k+1 - sum_k T_2k,j T_2k+1,j
 *           (half the vpmuludq; (xi + T) < 2^29 fits the 32-bit multiplier input; folded every 60 pairs)
 * Output: s_j mod b_j per message (scalar reduction, identical in both). */
#include <immintrin.h>
#include <stdint.h>
#define Q 4
void ext_plain(const uint64_t *xi /*[Q][n]*/, const uint64_t *T /*[n][n]*/, const uint64_t *b, int n, uint64_t *out /*[Q][n]*/) {
    for (int j = 0; j < n; j += 4) {
        __m256i acc[Q];
        for (int q = 0; q < Q; q++) acc[q] = _mm256_setzero_si256();
        for (int i = 0; i < n; i++) {
            __m256i t = _mm256_loadu_si256((const void *)(T + (size_t)i * n + j));
            for (int q = 0; q < Q; q++) acc[q] = _mm256_add_epi64(acc[q], _mm256_mul_epu32(_mm256_set1_epi64x(xi[q * n + i]), t));
        }
        for (int q = 0; q < Q; q++) { uint64_t v[4]; _mm256_storeu_si256((void *)v, acc[q]);
            for (int l = 0; l < 4; l++) out[q * n + j + l] = v[l] % b[j + l]; }
    }
}
void ext_wino(const uint64_t *xi, const uint64_t *T, const uint64_t *b, const uint64_t *CT /*[n] table half*/, int n, uint64_t *out) {
    unsigned __int128 CX[Q];
    for (int q = 0; q < Q; q++) { CX[q] = 0; for (int k = 0; k < n; k += 2) CX[q] += (unsigned __int128)xi[q * n + k] * xi[q * n + k + 1]; }
    for (int j = 0; j < n; j += 4) {
        uint64_t part[Q][4] = {{0}};
        for (int i0 = 0; i0 < n; i0 += 120) {
            int i1 = i0 + 120 < n ? i0 + 120 : n;
            __m256i acc[Q];
            for (int q = 0; q < Q; q++) acc[q] = _mm256_setzero_si256();
            for (int i = i0; i < i1; i += 2) {
                __m256i t0 = _mm256_loadu_si256((const void *)(T + (size_t)i * n + j)), t1 = _mm256_loadu_si256((const void *)(T + (size_t)(i + 1) * n + j));
                for (int q = 0; q < Q; q++) {
                    __m256i u = _mm256_add_epi64(_mm256_set1_epi64x(xi[q * n + i]), t1), w = _mm256_add_epi64(_mm256_set1_epi64x(xi[q * n + i + 1]), t0);
                    acc[q] = _mm256_add_epi64(acc[q], _mm256_mul_epu32(u, w));
                }
            }
            for (int q = 0; q < Q; q++) { uint64_t v[4]; _mm256_storeu_si256((void *)v, acc[q]);
                for (int l = 0; l < 4; l++) part[q][l] = (part[q][l] + v[l] % b[j + l]) % b[j + l]; }
        }
        for (int q = 0; q < Q; q++) for (int l = 0; l < 4; l++) {
            uint64_t m = b[j + l], cx = (uint64_t)(CX[q] % m);
            out[q * n + j + l] = (part[q][l] + 2 * m - cx - CT[j + l]) % m;
        }
    }
}
