# Provenance — Armilla Astrolabe 0.1.0

All code in this package was written for it from published method descriptions; no other implementation was read or copied.
No third-party code or data is vendored. Runtime dependencies: Python 3, NumPy; the kernel benchmark also uses gmpy2 (GMP) as a
baseline only.

| Idea | Source (credit) | Ours |
|---|---|---|
| Residue number system, CRT, mixed-radix conversion | Sunzi Suanjing; Garner 1959; Szabo & Tanaka 1967 | — |
| Fractional / approximate CRT for sign | Vu 1985; Kawamura, Koike, Sano & Shimbo 2000 | exact fallback band and its tests |
| Core function in RNS | Akushsky and co-workers 1977; Gonnella 1991 | — (the earlier unweighted sum is withdrawn) |
| Montgomery multiplication; RNS Montgomery | Montgomery 1985; Posch & Posch 1995; Bajard, Didier & Kornerup 1998; Kawamura et al. 2000 | mapping onto the six systems |
| Redundant-modulus base extension | Shenoy & Kumaresan 1989 | centre system as check channel |
| Lattice (gelosia) multiplication | Treviso Arithmetic 1478; Pacioli 1494 | ring residues in the boxes under the diagonal bound |
| Lazy modular reduction, limb (schoolbook) multiplication | field-known | — |
| Six-system layout (triangle, centre, top, bottom) | the author's design | ours |
| Astrolabe plate display | classical stereographic astrolabe | display only |

Prepared with AI assistance; the author reviews each document before publication.
