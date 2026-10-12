/* Evolved Gelosia lattice core (Armilla Astrolabe, 2026-10-12).
 * Every lattice box is one product per ring lane; products are summed along each diagonal
 * WITHOUT reducing (multiplication kept apart from accumulation, as in the paper Gelosia),
 * and each diagonal is reduced once per ring at the end. Lane-major layout: LANES rings side by side.
 * Residues < 256, products < 2^16, a diagonal of up to 64 boxes < 2^22: uint32 lanes never overflow. */
#include <stdint.h>
#define LANES 64
void gelosia_core(const uint32_t *RA, int KA, const uint32_t *RB, int KB,
                  const uint32_t *mods, uint32_t *D) {
    int nd = KA + KB - 1;
    for (int d = 0; d < nd; d++) {
        uint32_t acc[LANES] = {0};
        int i0 = d - KB + 1 > 0 ? d - KB + 1 : 0, i1 = d < KA - 1 ? d : KA - 1;
        for (int i = i0; i <= i1; i++) {
            const uint32_t *a = RA + i * LANES, *b = RB + (d - i) * LANES;
            for (int l = 0; l < LANES; l++) acc[l] += a[l] * b[l];   /* vectorised by the compiler */
        }
        /* one reduction per diagonal, vectorised: q from a float reciprocal (acc < 2^22 is exact in float),
         * then at most one correction step each way */
        for (int l = 0; l < LANES; l++) {
            float inv = 1.0f / (float)mods[l];
            int32_t r = (int32_t)acc[l] - (int32_t)((float)acc[l] * inv) * (int32_t)mods[l];
            r += (r < 0) ? (int32_t)mods[l] : 0;
            r -= (r >= (int32_t)mods[l]) ? (int32_t)mods[l] : 0;
            D[d * LANES + l] = (uint32_t)r;
        }
    }
}
