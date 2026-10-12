/* AVX2-only pull engine for legacy CPUs (2026-10-12): 2-trit codes (9 patterns), answers pulled with pshufb (16-entry byte table
 * per 128-bit lane), 3 data rings (251, 241, 239), 16-bit lazy sums, vector Barrett every 128 groups, L1 table tiles.
 * Baseline: omnid's production form (rns_gemv_bench tdot: sign_epi8 -> madd -> int32), one row at a time. */
#include <immintrin.h>
#include <stdint.h>
#define R 3
static const int MODS[R] = {251, 241, 239};

/* x int8[K] -> lut uint8[G2][R][32] (the 9 answers of each pair of inputs, the 16-byte table repeated in both lanes) */
void pull2_precompute(const int8_t *x, int K, uint8_t *lut) {
    int G2 = K / 2;
    for (int j = 0; j < G2; j++)
        for (int r = 0; r < R; r++) {
            uint8_t *t = lut + ((size_t)j * R + r) * 32;
            for (int p = 0; p < 16; p++) {
                int s = p < 9 ? (p % 3 - 1) * x[2 * j] + (p / 3 - 1) * x[2 * j + 1] : 0;
                s %= MODS[r]; if (s < 0) s += MODS[r];
                t[p] = t[16 + p] = (uint8_t)s;
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
/* codes uint8[G2][O] (O % 32 == 0) -> out uint8[R][O] */
void pull2_gemv(const uint8_t *codes, const uint8_t *lut, int G2, int O, uint8_t *out) {
    for (int ob = 0; ob < O; ob += 32) {
        __m256i lo[R], hi[R];
        for (int r = 0; r < R; r++) { lo[r] = _mm256_setzero_si256(); hi[r] = _mm256_setzero_si256(); }
        int cnt = 0;
        for (int j = 0; j < G2; j++) {
            __m256i idx = _mm256_loadu_si256((const void *)(codes + (size_t)j * O + ob));
            const uint8_t *T = lut + (size_t)j * R * 32;
            for (int r = 0; r < R; r++) {
                __m256i v = _mm256_shuffle_epi8(_mm256_loadu_si256((const void *)(T + 32 * r)), idx);   /* the pull */
                lo[r] = _mm256_add_epi16(lo[r], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v)));
                hi[r] = _mm256_add_epi16(hi[r], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1)));
            }
            if (++cnt == 128) { for (int r = 0; r < R; r++) { lo[r] = barrett16(lo[r], MODS[r]); hi[r] = barrett16(hi[r], MODS[r]); } cnt = 0; }
        }
        for (int r = 0; r < R; r++) {
            uint16_t a[16], b[16];
            _mm256_storeu_si256((void *)a, barrett16(lo[r], MODS[r])); _mm256_storeu_si256((void *)b, barrett16(hi[r], MODS[r]));
            for (int l = 0; l < 16; l++) { out[(size_t)r * O + ob + l] = (uint8_t)a[l]; out[(size_t)r * O + ob + 16 + l] = (uint8_t)b[l]; }
        }
    }
}
/* omnid production form: tdot per row */
static inline int32_t tdot(const int8_t *x, const int8_t *t, int n) {
    __m256i acc = _mm256_setzero_si256(); const __m256i ones = _mm256_set1_epi16(1);
    for (int k = 0; k < n; k += 32) {
        __m256i v = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(x + k)), _mm256_loadu_si256((const __m256i *)(t + k)));
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)), ones),
                                                      _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)), ones)));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s); return _mm_cvtsi128_si32(s);
}
void gemv_sign(const int8_t *W, const int8_t *x, int O, int K, int32_t *y) { for (int o = 0; o < O; o++) y[o] = tdot(x, W + (size_t)o * K, K); }

/* v2: codes stored row-block-major, uint8[O/32][G2][32] (contiguous stream per block), tables tiled so a tile stays in L1,
 * partial sums kept in acc16 uint16[O/32][R][32] between tiles. */
void pull2_gemv_blk(const uint8_t *codesb, const uint8_t *lut, int G2, int O, uint8_t *out, uint16_t *acc16, int JT) {
    int NB = O / 32;
    for (size_t i = 0; i < (size_t)NB * R * 32; i++) acc16[i] = 0;
    for (int j0 = 0; j0 < G2; j0 += JT) {
        int j1 = j0 + JT < G2 ? j0 + JT : G2;
        for (int bl = 0; bl < NB; bl++) {
            __m256i lo[R], hi[R];
            for (int r = 0; r < R; r++) { lo[r] = _mm256_loadu_si256((const void *)(acc16 + ((size_t)bl * R + r) * 32));
                                          hi[r] = _mm256_loadu_si256((const void *)(acc16 + ((size_t)bl * R + r) * 32 + 16)); }
            const uint8_t *cb = codesb + (size_t)bl * G2 * 32;
            for (int j = j0; j < j1; j++) {
                __m256i idx = _mm256_loadu_si256((const void *)(cb + (size_t)j * 32));
                const uint8_t *T = lut + (size_t)j * R * 32;
                for (int r = 0; r < R; r++) {
                    __m256i v = _mm256_shuffle_epi8(_mm256_loadu_si256((const void *)(T + 32 * r)), idx);
                    lo[r] = _mm256_add_epi16(lo[r], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v)));
                    hi[r] = _mm256_add_epi16(hi[r], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1)));
                }
            }
            for (int r = 0; r < R; r++) {
                _mm256_storeu_si256((void *)(acc16 + ((size_t)bl * R + r) * 32), barrett16(lo[r], MODS[r]));
                _mm256_storeu_si256((void *)(acc16 + ((size_t)bl * R + r) * 32 + 16), barrett16(hi[r], MODS[r]));
            }
        }
    }
    for (int bl = 0; bl < NB; bl++) for (int r = 0; r < R; r++) for (int l = 0; l < 32; l++)
        out[(size_t)r * O + bl * 32 + l] = (uint8_t)acc16[((size_t)bl * R + r) * 32 + l];
}
