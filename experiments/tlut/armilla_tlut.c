/* Armilla ternary pull engine (2026-10-12).
 *
 * The vision, as built here:
 *   - ternary weights {-1, 0, +1}: no multiplies anywhere;
 *   - OUTER RINGS = PRECOMPUTERS: for each group of 3 inputs and each ring m, all 27 possible ternary answers
 *     (w0*x0 + w1*x1 + w2*x2) mod m are computed once per input vector and laid out as a 64-byte table;
 *   - the CENTRE pulls the answer instead of computing it: every weight row stores, per group, which of the 27 answers
 *     it needs (one byte, 3 weights). One AVX-512 VBMI vpermb pulls the answers of 64 rows at once from the table;
 *   - answers are summed in the ring with no carries between rings (lazy: reduced once every 256 groups);
 *   - the read-out (fractional CRT) and ReLU / requantisation are done once per output (see tlut_bench.py).
 */
#include <immintrin.h>
#include <stdint.h>

/* precompute: x int8[K] (K = 3*G), mods[R] -> lut[R][G][64] */
void tlut_precompute(const int8_t *x, int G, const uint16_t *mods, int R, uint8_t *lut) {
    for (int r = 0; r < R; r++) {
        int m = mods[r];
        for (int j = 0; j < G; j++) {
            int x0 = x[3 * j], x1 = x[3 * j + 1], x2 = x[3 * j + 2];
            uint8_t *t = lut + ((size_t)r * G + j) * 64;
            for (int p = 0; p < 27; p++) {
                int s = (p % 3 - 1) * x0 + ((p / 3) % 3 - 1) * x1 + (p / 9 - 1) * x2;
                s %= m; if (s < 0) s += m;
                t[p] = (uint8_t)s;
            }
            for (int p = 27; p < 64; p++) t[p] = 0;
        }
    }
}

static inline void reduce16(__m512i *acc, int m) {
    uint16_t tmp[32];
    _mm512_storeu_si512(tmp, *acc);
    for (int l = 0; l < 32; l++) tmp[l] %= m;
    *acc = _mm512_loadu_si512(tmp);
}

/* centre: codes uint8[G][O] (group-major, O % 64 == 0), lut[R][G][64] -> out uint8[R][O] (residues) */
void tlut_gemv(const uint8_t *codes, const uint8_t *lut, int R, int G, int O, const uint16_t *mods, uint8_t *out) {
    for (int r = 0; r < R; r++) {
        int m = mods[r];
        const uint8_t *L = lut + (size_t)r * G * 64;
        for (int ob = 0; ob < O; ob += 64) {
            __m512i lo = _mm512_setzero_si512(), hi = _mm512_setzero_si512();
            int cnt = 0;
            for (int j = 0; j < G; j++) {
                __m512i idx = _mm512_loadu_si512(codes + (size_t)j * O + ob);
                __m512i tab = _mm512_loadu_si512(L + (size_t)j * 64);
                __m512i v = _mm512_permutexvar_epi8(idx, tab);            /* the pull: 64 answers, one instruction */
                lo = _mm512_add_epi16(lo, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v)));
                hi = _mm512_add_epi16(hi, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(v, 1)));
                if (++cnt == 256) { reduce16(&lo, m); reduce16(&hi, m); cnt = 0; }
            }
            uint16_t a[32], b[32];
            _mm512_storeu_si512(a, lo); _mm512_storeu_si512(b, hi);
            for (int l = 0; l < 32; l++) { out[(size_t)r * O + ob + l] = a[l] % m; out[(size_t)r * O + ob + 32 + l] = b[l] % m; }
        }
    }
}

/* baseline 1: int8 weights x int8 inputs, int32 accumulate (compiler-vectorised, AVX-512) */
void gemv_i8(const int8_t *W, const int8_t *x, int O, int K, int32_t *y) {
    for (int o = 0; o < O; o++) {
        int32_t s = 0;
        const int8_t *w = W + (size_t)o * K;
        for (int k = 0; k < K; k++) s += (int32_t)w[k] * (int32_t)x[k];
        y[o] = s;
    }
}

/* baseline 2: ternary weights as two bit-planes (plus / minus), int8 inputs: masked adds, no multiplies */
void gemv_bitplane(const uint64_t *plus, const uint64_t *minus, const int8_t *x, int O, int K, int32_t *y) {
    int words = K / 64;
    for (int o = 0; o < O; o++) {
        __m512i acc = _mm512_setzero_si512();
        for (int wv = 0; wv < words; wv++) {
            __m512i xv = _mm512_loadu_si512(x + wv * 64);
            __mmask64 pm = plus[(size_t)o * words + wv], mm = minus[(size_t)o * words + wv];
            __m512i a = _mm512_maskz_mov_epi8(pm, xv), s = _mm512_maskz_mov_epi8(mm, xv);
            /* widen to 16-bit and add/subtract */
            acc = _mm512_add_epi16(acc, _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a)));
            acc = _mm512_add_epi16(acc, _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a, 1)));
            acc = _mm512_sub_epi16(acc, _mm512_cvtepi8_epi16(_mm512_castsi512_si256(s)));
            acc = _mm512_sub_epi16(acc, _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(s, 1)));
            if ((wv & 7) == 7) { /* keep 16-bit lanes safe: fold into 32-bit every 8 words */
                __m512i l32 = _mm512_madd_epi16(acc, _mm512_set1_epi16(1));
                y[o] += _mm512_reduce_add_epi32(l32);
                acc = _mm512_setzero_si512();
            }
        }
        __m512i l32 = _mm512_madd_epi16(acc, _mm512_set1_epi16(1));
        y[o] += _mm512_reduce_add_epi32(l32);
    }
}

/* ---------------- v2: all rings turn together, two groups per table ----------------
 * codes2 uint8[G/2][O][2]: for row o and group pair q the bytes (code(2q), code(2q+1) + 32).
 * lut2  uint8[G/2][R][64]: entries 0..26 = answers of group 2q, 32..58 = answers of group 2q+1 (per ring).
 * One load of 64 code bytes (32 rows x 2 groups) feeds every ring; one vpermb per ring pulls 64 answers;
 * vpmaddubsw adds each row's two answers in one instruction; reduction is vector Barrett every 128 pairs. */
void tlut_precompute2(const int8_t *x, int G, const uint16_t *mods, int R, uint8_t *lut2) {
    for (int q = 0; q < G / 2; q++)
        for (int r = 0; r < R; r++) {
            int m = mods[r];
            uint8_t *t = lut2 + ((size_t)q * R + r) * 64;
            for (int h = 0; h < 2; h++) {
                int j = 2 * q + h, x0 = x[3 * j], x1 = x[3 * j + 1], x2 = x[3 * j + 2];
                for (int p = 0; p < 27; p++) {
                    int s = (p % 3 - 1) * x0 + ((p / 3) % 3 - 1) * x1 + (p / 9 - 1) * x2;
                    s %= m; if (s < 0) s += m;
                    t[32 * h + p] = (uint8_t)s;
                }
                for (int p = 27; p < 32; p++) t[32 * h + p] = 0;
            }
        }
}

static inline __m512i barrett16(__m512i a, int m) {
    __m512i q = _mm512_mulhi_epu16(a, _mm512_set1_epi16((short)(65536 / m)));
    __m512i r = _mm512_sub_epi16(a, _mm512_mullo_epi16(q, _mm512_set1_epi16((short)m)));
    __m512i mm = _mm512_set1_epi16((short)m);
    r = _mm512_min_epu16(r, _mm512_sub_epi16(r, mm));
    r = _mm512_min_epu16(r, _mm512_sub_epi16(r, mm));
    return r;
}

#define RMAX 8
void tlut_gemv2(const uint8_t *codes2, const uint8_t *lut2, int R, int G, int O, const uint16_t *mods, uint8_t *out) {
    const __m512i ones = _mm512_set1_epi8(1);
    int Q = G / 2;
    for (int ob = 0; ob < O; ob += 32) {
        __m512i acc[RMAX];
        for (int r = 0; r < R; r++) acc[r] = _mm512_setzero_si512();
        int cnt = 0;
        for (int q = 0; q < Q; q++) {
            __m512i idx = _mm512_loadu_si512(codes2 + ((size_t)q * O + ob) * 2);
            const uint8_t *T = lut2 + (size_t)q * R * 64;
            for (int r = 0; r < R; r++) {
                __m512i v = _mm512_permutexvar_epi8(idx, _mm512_loadu_si512(T + r * 64));
                acc[r] = _mm512_add_epi16(acc[r], _mm512_maddubs_epi16(v, ones));
            }
            if (++cnt == 128) { for (int r = 0; r < R; r++) acc[r] = barrett16(acc[r], mods[r]); cnt = 0; }
        }
        for (int r = 0; r < R; r++) {
            uint16_t a[32];
            _mm512_storeu_si512(a, barrett16(acc[r], mods[r]));
            for (int l = 0; l < 32; l++) out[(size_t)r * O + ob + l] = (uint8_t)a[l];
        }
    }
}

/* ---------------- v3: v2 + L1-resident table tiles ----------------
 * The pair tables of QT group pairs (QT * R * 64 bytes, 24 KB at QT = 96, R = 4) stay in L1 while every row block pulls from
 * them; per-row partial sums live in acc16 uint16[O/32][R][32] between tiles (each tile adds < 96 * 500 < 65536, then Barrett). */
void tlut_gemv3(const uint8_t *codes2, const uint8_t *lut2, int R, int G, int O, const uint16_t *mods, uint8_t *out,
                uint16_t *acc16, int QT) {
    const __m512i ones = _mm512_set1_epi8(1);
    int Q = G / 2, NB = O / 32;
    for (size_t i = 0; i < (size_t)NB * R * 32; i++) acc16[i] = 0;
    for (int q0 = 0; q0 < Q; q0 += QT) {
        int q1 = q0 + QT < Q ? q0 + QT : Q;
        for (int bl = 0; bl < NB; bl++) {
            int ob = bl * 32;
            __m512i acc[RMAX];
            for (int r = 0; r < R; r++) acc[r] = _mm512_loadu_si512(acc16 + ((size_t)bl * R + r) * 32);
            for (int q = q0; q < q1; q++) {
                __m512i idx = _mm512_loadu_si512(codes2 + ((size_t)q * O + ob) * 2);
                const uint8_t *T = lut2 + (size_t)q * R * 64;
                for (int r = 0; r < R; r++) {
                    __m512i v = _mm512_permutexvar_epi8(idx, _mm512_loadu_si512(T + r * 64));
                    acc[r] = _mm512_add_epi16(acc[r], _mm512_maddubs_epi16(v, ones));
                }
            }
            for (int r = 0; r < R; r++) _mm512_storeu_si512(acc16 + ((size_t)bl * R + r) * 32, barrett16(acc[r], mods[r]));
        }
    }
    for (int bl = 0; bl < NB; bl++)
        for (int r = 0; r < R; r++)
            for (int l = 0; l < 32; l++) out[(size_t)r * O + bl * 32 + l] = (uint8_t)acc16[((size_t)bl * R + r) * 32 + l];
}

/* ---------------- vectorised precompute (the outer rings) ----------------
 * For each group: the 27 answers c0*x0 + c1*x1 + c2*x2 in 32 int16 lanes (pad lanes use the all-zero pattern), shifted by 2m and
 * reduced by three conditional subtractions (|s| <= 381 < 2m for m >= 191). Group 2q fills bytes 0..31, group 2q+1 bytes 32..63
 * of the pair table: the same layout as tlut_precompute2. */
void tlut_precompute3(const int8_t *x, int G, const uint16_t *mods, int R, uint8_t *lut2) {
    int16_t c0[32], c1[32], c2[32];
    for (int p = 0; p < 32; p++) {
        int pp = p < 27 ? p : 13;
        c0[p] = pp % 3 - 1; c1[p] = (pp / 3) % 3 - 1; c2[p] = pp / 9 - 1;
    }
    __m512i C0 = _mm512_loadu_si512(c0), C1 = _mm512_loadu_si512(c1), C2 = _mm512_loadu_si512(c2);
    for (int q = 0; q < G / 2; q++) {
        for (int h = 0; h < 2; h++) {
            const int8_t *a = x + 6 * q + 3 * h;
            __m512i S = _mm512_add_epi16(_mm512_add_epi16(_mm512_mullo_epi16(C0, _mm512_set1_epi16(a[0])),
                                                          _mm512_mullo_epi16(C1, _mm512_set1_epi16(a[1]))),
                                         _mm512_mullo_epi16(C2, _mm512_set1_epi16(a[2])));
            for (int r = 0; r < R; r++) {
                __m512i mm = _mm512_set1_epi16((short)mods[r]);
                __m512i v = _mm512_add_epi16(S, _mm512_add_epi16(mm, mm));
                v = _mm512_min_epu16(v, _mm512_sub_epi16(v, mm));
                v = _mm512_min_epu16(v, _mm512_sub_epi16(v, mm));
                v = _mm512_min_epu16(v, _mm512_sub_epi16(v, mm));
                _mm256_storeu_si256((void *)(lut2 + ((size_t)q * R + r) * 64 + 32 * h), _mm512_cvtepi16_epi8(v));
            }
        }
    }
}

/* ---------------- Gelosia digits: the outer rings fill one table set per (activation digit, ring) ----------------
 * xd int8[D][K]: balanced base-256 digits of the activations (x = sum_d xd[d] * 256^d). lut uint8[G/2][D*R][64],
 * table index t = d*R + r. Same answers and layout as tlut_precompute3, one set per digit. */
void tlut_precompute_digits(const int8_t *xd, int K, int D, const uint16_t *mods, int R, uint8_t *lut) {
    int G = K / 3, RT = D * R;
    int16_t c0[32], c1[32], c2[32];
    for (int p = 0; p < 32; p++) { int pp = p < 27 ? p : 13; c0[p] = pp % 3 - 1; c1[p] = (pp / 3) % 3 - 1; c2[p] = pp / 9 - 1; }
    __m512i C0 = _mm512_loadu_si512(c0), C1 = _mm512_loadu_si512(c1), C2 = _mm512_loadu_si512(c2);
    for (int q = 0; q < G / 2; q++)
        for (int d = 0; d < D; d++)
            for (int h = 0; h < 2; h++) {
                const int8_t *a = xd + (size_t)d * K + 6 * q + 3 * h;
                __m512i S = _mm512_add_epi16(_mm512_add_epi16(_mm512_mullo_epi16(C0, _mm512_set1_epi16(a[0])),
                                                              _mm512_mullo_epi16(C1, _mm512_set1_epi16(a[1]))),
                                             _mm512_mullo_epi16(C2, _mm512_set1_epi16(a[2])));
                for (int r = 0; r < R; r++) {
                    __m512i mm = _mm512_set1_epi16((short)mods[r]);
                    __m512i v = _mm512_add_epi16(S, _mm512_add_epi16(mm, mm));
                    v = _mm512_min_epu16(v, _mm512_sub_epi16(v, mm));
                    v = _mm512_min_epu16(v, _mm512_sub_epi16(v, mm));
                    v = _mm512_min_epu16(v, _mm512_sub_epi16(v, mm));
                    _mm256_storeu_si256((void *)(lut + ((size_t)q * RT + d * R + r) * 64 + 32 * h), _mm512_cvtepi16_epi8(v));
                }
            }
}

/* ---------------- 5-trit codes (THEAD P3 digit order: code = sum_j (w_j + 1) * 3^j, 243 patterns, 1.6 bits / weight) --------
 * lut5 uint8[G5][R][256]: per group of 5 inputs and ring, the 243 answers (sum_j w_j x_j) mod m (entries 243..255 unused).
 * codes5 uint8[G5][O]: one byte per row per group. The pull: two vpermi2b (128-entry halves) + one blend on bit 7 of the code
 * pull 64 rows' answers; 16-bit widening adds; L1 tiles of 24 groups; vector Barrett per tile. */
void tlut5_precompute(const int8_t *x, int G5, const uint16_t *mods, int R, uint8_t *lut5) {
    static int16_t C[5][256]; static int init = 0;
    if (!init) { for (int p = 0; p < 256; p++) { int v = p < 243 ? p : 121; for (int j = 0; j < 5; j++) { C[j][p] = v % 3 - 1; v /= 3; } } init = 1; }
    for (int q = 0; q < G5; q++) {
        const int8_t *a = x + 5 * q;
        for (int z = 0; z < 8; z++) {                       /* 8 vectors x 32 lanes = 256 patterns */
            __m512i S = _mm512_setzero_si512();
            for (int j = 0; j < 5; j++)
                S = _mm512_add_epi16(S, _mm512_mullo_epi16(_mm512_loadu_si512(&C[j][32 * z]), _mm512_set1_epi16(a[j])));
            for (int r = 0; r < R; r++) {
                __m512i mm = _mm512_set1_epi16((short)mods[r]);
                __m512i v = _mm512_add_epi16(S, _mm512_mullo_epi16(mm, _mm512_set1_epi16(3)));   /* |S| <= 635 < 3m */
                for (int k = 0; k < 5; k++) v = _mm512_min_epu16(v, _mm512_sub_epi16(v, mm));
                _mm256_storeu_si256((void *)(lut5 + ((size_t)q * R + r) * 256 + 32 * z), _mm512_cvtepi16_epi8(v));
            }
        }
    }
}

void tlut5_gemv(const uint8_t *codes5, const uint8_t *lut5, int R, int G5, int O, const uint16_t *mods, uint8_t *out,
                uint16_t *acc16, int QT) {
    int NB = O / 64;
    for (size_t i = 0; i < (size_t)NB * R * 64; i++) acc16[i] = 0;
    for (int q0 = 0; q0 < G5; q0 += QT) {
        int q1 = q0 + QT < G5 ? q0 + QT : G5;
        for (int bl = 0; bl < NB; bl++) {
            __m512i lo[RMAX], hi[RMAX];
            for (int r = 0; r < R; r++) { lo[r] = _mm512_loadu_si512(acc16 + ((size_t)bl * R + r) * 64);
                                          hi[r] = _mm512_loadu_si512(acc16 + ((size_t)bl * R + r) * 64 + 32); }
            for (int q = q0; q < q1; q++) {
                __m512i idx = _mm512_loadu_si512(codes5 + (size_t)q * O + bl * 64);
                __mmask64 top = _mm512_movepi8_mask(idx);
                const uint8_t *T = lut5 + (size_t)q * R * 256;
                for (int r = 0; r < R; r++) {
                    const uint8_t *t = T + r * 256;
                    __m512i a = _mm512_permutex2var_epi8(_mm512_loadu_si512(t), idx, _mm512_loadu_si512(t + 64));
                    __m512i b = _mm512_permutex2var_epi8(_mm512_loadu_si512(t + 128), idx, _mm512_loadu_si512(t + 192));
                    __m512i v = _mm512_mask_blend_epi8(top, a, b);
                    lo[r] = _mm512_add_epi16(lo[r], _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v)));
                    hi[r] = _mm512_add_epi16(hi[r], _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(v, 1)));
                }
            }
            for (int r = 0; r < R; r++) {
                _mm512_storeu_si512(acc16 + ((size_t)bl * R + r) * 64, barrett16(lo[r], mods[r]));
                _mm512_storeu_si512(acc16 + ((size_t)bl * R + r) * 64 + 32, barrett16(hi[r], mods[r]));
            }
        }
    }
    for (int bl = 0; bl < NB; bl++)
        for (int r = 0; r < R; r++)
            for (int l = 0; l < 64; l++) out[(size_t)r * O + bl * 64 + l] = (uint8_t)acc16[((size_t)bl * R + r) * 64 + l];
}

/* ---------------- forward with a batch of tokens: one code load feeds NT tokens x R rings (prefill / batched decode) --------
 * lutb uint8[G/2][NT][R][64] (tlut_precompute3 run per token into its slot); outb uint8[NT][R][O]. NT * R <= 24 accumulators. */
#define NTMAX 8
void tlut_gemv3_batch(const uint8_t *codes2, const uint8_t *lutb, int NT, int R, int G, int O, const uint16_t *mods,
                      uint8_t *outb, uint16_t *acc16, int QT) {
    const __m512i ones = _mm512_set1_epi8(1);
    int Q = G / 2, NB = O / 32, TR = NT * R;
    for (size_t i = 0; i < (size_t)NB * TR * 32; i++) acc16[i] = 0;
    for (int q0 = 0; q0 < Q; q0 += QT) {
        int q1 = q0 + QT < Q ? q0 + QT : Q;
        for (int bl = 0; bl < NB; bl++) {
            __m512i acc[NTMAX * 3];
            for (int k = 0; k < TR; k++) acc[k] = _mm512_loadu_si512(acc16 + ((size_t)bl * TR + k) * 32);
            for (int q = q0; q < q1; q++) {
                __m512i idx = _mm512_loadu_si512(codes2 + ((size_t)q * O + bl * 32) * 2);
                const uint8_t *T = lutb + (size_t)q * TR * 64;
                for (int k = 0; k < TR; k++)
                    acc[k] = _mm512_add_epi16(acc[k], _mm512_maddubs_epi16(_mm512_permutexvar_epi8(idx, _mm512_loadu_si512(T + k * 64)), ones));
            }
            for (int k = 0; k < TR; k++) _mm512_storeu_si512(acc16 + ((size_t)bl * TR + k) * 32, barrett16(acc[k], mods[k % R]));
        }
    }
    for (int bl = 0; bl < NB; bl++)
        for (int k = 0; k < TR; k++)
            for (int l = 0; l < 32; l++) outb[(size_t)k * O + bl * 32 + l] = (uint8_t)acc16[((size_t)bl * TR + k) * 32 + l];
}

/* batched bit-plane baseline: masks loaded once per word, applied to NT token vectors */
void gemv_bitplane_batch(const uint64_t *plus, const uint64_t *minus, const int8_t *X /*[NT][K]*/, int NT, int O, int K, int32_t *Y /*[NT][O]*/) {
    int words = K / 64;
    for (int o = 0; o < O; o++) {
        __m512i acc[NTMAX];
        for (int t = 0; t < NT; t++) acc[t] = _mm512_setzero_si512();
        for (int wv = 0; wv < words; wv++) {
            __mmask64 pm = plus[(size_t)o * words + wv], mm = minus[(size_t)o * words + wv];
            for (int t = 0; t < NT; t++) {
                __m512i xv = _mm512_loadu_si512(X + (size_t)t * K + wv * 64);
                __m512i a = _mm512_maskz_mov_epi8(pm, xv), s = _mm512_maskz_mov_epi8(mm, xv);
                /* sum bytes into 16-bit pairs: maddubs needs unsigned x signed -> use (a - s) as signed and ones as unsigned */
                __m512i d = _mm512_sub_epi8(a, s);
                acc[t] = _mm512_add_epi16(acc[t], _mm512_maddubs_epi16(_mm512_set1_epi8(1), d));
            }
            if ((wv & 31) == 31) for (int t = 0; t < NT; t++) {
                Y[(size_t)t * O + o] += _mm512_reduce_add_epi32(_mm512_madd_epi16(acc[t], _mm512_set1_epi16(1))); acc[t] = _mm512_setzero_si512(); }
        }
        for (int t = 0; t < NT; t++) Y[(size_t)t * O + o] += _mm512_reduce_add_epi32(_mm512_madd_epi16(acc[t], _mm512_set1_epi16(1)));
    }
}

/* ---------------- training step helpers: carry-free ring all-reduce of int32 gradient partials, exact Garner read-out ---------
 * acc[r][i] = (acc[r][i] + p[i] mod m_r) mod m_r for the 4 rings 251, 241, 239, 233 (literal moduli: the compiler turns every %
 * into a multiply-shift and vectorises). Any harness order gives the same bytes. */
#define RING_ACC(M, A) { for (size_t i = 0; i < n; i++) { int32_t v = p[i] % M; v += v < 0 ? M : 0; int32_t s = A[i] + v; A[i] = (uint8_t)(s >= M ? s - M : s); } }
void ring_acc4(const int32_t *p, uint8_t *a0, uint8_t *a1, uint8_t *a2, uint8_t *a3, size_t n) {
    RING_ACC(251, a0) RING_ACC(241, a1) RING_ACC(239, a2) RING_ACC(233, a3)
}
/* Garner over (251, 241, 239, 233) -> signed int64, centred */
void garner4(const uint8_t *a0, const uint8_t *a1, const uint8_t *a2, const uint8_t *a3, int64_t *out, size_t n) {
    /* inverses: 251^-1 mod 241 = 159?, computed at run time below to stay exact */
    static int inv[4][4]; static int init = 0; const int m[4] = {251, 241, 239, 233};
    if (!init) { for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { inv[i][j] = 0; for (int x = 1; x < m[i]; x++) if ((m[j] % m[i]) * x % m[i] == 1) { inv[i][j] = x; break; } } init = 1; }
    const int64_t M = 251LL * 241 * 239 * 233;
    for (size_t k = 0; k < n; k++) {
        int r[4] = {a0[k], a1[k], a2[k], a3[k]}, d[4];
        for (int i = 0; i < 4; i++) { int t = r[i]; for (int j = 0; j < i; j++) t = ((t - d[j]) % m[i] + m[i]) % m[i] * inv[i][j] % m[i]; d[i] = t; }
        int64_t y = d[0] + (int64_t)251 * (d[1] + (int64_t)241 * (d[2] + (int64_t)239 * d[3]));
        out[k] = 2 * y >= M ? y - M : y;
    }
}

/* 5-trit pull, row-block-major codes: codes5b uint8[O/64][G5][64] (one contiguous stream per 64-row block) */
void tlut5_gemv_blk(const uint8_t *codes5b, const uint8_t *lut5, int R, int G5, int O, const uint16_t *mods, uint8_t *out,
                    uint16_t *acc16, int QT) {
    int NB = O / 64;
    for (size_t i = 0; i < (size_t)NB * R * 64; i++) acc16[i] = 0;
    for (int q0 = 0; q0 < G5; q0 += QT) {
        int q1 = q0 + QT < G5 ? q0 + QT : G5;
        for (int bl = 0; bl < NB; bl++) {
            __m512i lo[RMAX], hi[RMAX];
            for (int r = 0; r < R; r++) { lo[r] = _mm512_loadu_si512(acc16 + ((size_t)bl * R + r) * 64);
                                          hi[r] = _mm512_loadu_si512(acc16 + ((size_t)bl * R + r) * 64 + 32); }
            const uint8_t *cb = codes5b + (size_t)bl * G5 * 64;
            for (int q = q0; q < q1; q++) {
                __m512i idx = _mm512_loadu_si512(cb + (size_t)q * 64);
                __mmask64 top = _mm512_movepi8_mask(idx);
                const uint8_t *T = lut5 + (size_t)q * R * 256;
                for (int r = 0; r < R; r++) {
                    const uint8_t *t = T + r * 256;
                    __m512i a = _mm512_permutex2var_epi8(_mm512_loadu_si512(t), idx, _mm512_loadu_si512(t + 64));
                    __m512i b = _mm512_permutex2var_epi8(_mm512_loadu_si512(t + 128), idx, _mm512_loadu_si512(t + 192));
                    __m512i v = _mm512_mask_blend_epi8(top, a, b);
                    lo[r] = _mm512_add_epi16(lo[r], _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v)));
                    hi[r] = _mm512_add_epi16(hi[r], _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(v, 1)));
                }
            }
            for (int r = 0; r < R; r++) {
                _mm512_storeu_si512(acc16 + ((size_t)bl * R + r) * 64, barrett16(lo[r], mods[r]));
                _mm512_storeu_si512(acc16 + ((size_t)bl * R + r) * 64 + 32, barrett16(hi[r], mods[r]));
            }
        }
    }
    for (int bl = 0; bl < NB; bl++) for (int r = 0; r < R; r++) for (int l = 0; l < 64; l++)
        out[(size_t)r * O + bl * 64 + l] = (uint8_t)acc16[((size_t)bl * R + r) * 64 + l];
}
