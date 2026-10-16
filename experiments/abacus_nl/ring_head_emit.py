#!/usr/bin/env python3
"""ABACUS-NL-1 step 3: emit ring_head.h — the ring softmax/LSE/loss primitives (jit_nl.render_prims) as host+device code for the CUDA trainer
(thead-trial/gpu/emit/cuda.py includes it), plus the per-element term and the per-row tail the two trainer kernels (rrz, hstatsR) call.
usage: ring_head_emit.py OUT.h   (deterministic)"""
import re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jit_nl as J
RN = J.RN
from ring_nl import MODS, S, ZMAX, KMAX, EXP_D, ATANH_J, NEWTON
FAST = os.environ.get('RING_HEAD_FAST', '1') == '1'            # ABACUS-NL-2 step 1: fused/dot-product SCALE (bit-identical); RING_HEAD_FAST=0 re-emits the NL-1 header
prims = J.render_prims(cuda=True, fast=FAST, sub=FAST)
lines = []
for l in prims.split("\n"):
    if l.startswith(("#include", "#define CONST", "#define DEV", "typedef uint32_t")): continue
    m = re.match(r"CONST u32 (\w+)(\[.*)", l)
    if m:                                   # a table: a host copy and a __constant__ device copy; the name picks by compile pass
        n, rest = m.groups()
        lines.append(f"static const u32 H_{n}{rest}\n#ifdef OM_CUDA\n__constant__ u32 D_{n}{rest}\n#endif\n"
                     f"#if defined(__CUDA_ARCH__)\n#define {n} D_{n}\n#else\n#define {n} H_{n}\n#endif")
    else: lines.append(l)
T2G = "{" + ", ".join("{" + ", ".join(map(str, J.res(c))) + "}" for c in J.RN.T2) + "}"
mod_tab = "{" + ", ".join(f"{m}u" for m in MODS) + "}"
tail = r"""
#ifdef __CUDACC__
#define RH_NOINL __host__ __device__ __noinline__
#else
#define RH_NOINL static __attribute__((noinline))
#endif
// PULL table T2 for the per-element lookup: the index is data-dependent (differs per lane), and __constant__ reads serialise across a warp when
// the index diverges, so the device reads a plain global copy (L1-cached); the host reads the same table from H_C_T2
#ifdef OM_CUDA
__device__ u32 D_T2G[ZMAXV + 1][NM] = T2GTAB;
#endif
#if defined(__CUDA_ARCH__)
#define RH_T2ROW(z) (D_T2G[(z)])
#else
#define RH_T2ROW(z) (H_C_T2[(z)])
#endif
static const u32 H_RH_MOD[NM] = MODTAB;
#ifdef OM_CUDA
__constant__ u32 D_RH_MOD[NM] = MODTAB;
#endif
#if defined(__CUDA_ARCH__)
#define RH_MOD D_RH_MOD
#else
#define RH_MOD H_RH_MOD
#endif
// quantiser: a float logit -> its integer at scale S (double product, one rounding; no FMA: --fmad=false / -ffp-contract=off)
OM_HD i64 rh_q(float l) { return (i64)floor((double)l * SVAL + 0.5); }
// one element's term: w = (qmx - q) as residues -> exp mantissa -> times the PULL table entry; returns the exponent z (or -1: too large, term 0)
RH_NOINL int rh_term(u32*t, i64 d) {
  u32 w[NM],q[NM],z[NM],tt[NM],rr[NM],nr[NM],acc[NM];
  v_set_i64(w,d);
  v_mul(q,w,C_INVLN2); v_scale(q,q); v_scale(z,q);
  v_mul(tt,z,C_LN2_2); v_scale(tt,tt); v_sub(rr,w,tt); v_neg(nr,rr);
  memcpy(acc,C_EXP[EXPD],sizeof(acc));
  for(int k=EXPD-1;k>=0;k--){ v_mul(acc,acc,nr); v_scale(acc,acc); v_add(acc,acc,C_EXP[k]); }
  i64 zi=v_small(z); if(zi<0) return -1;
  v_mul(t,acc,RH_T2ROW(zi>ZMAXV?ZMAXV:zi)); return (int)zi;
}
@@INTTERM@@
RH_NOINL void rh_newton(u32*y,const u32*t,const u32*y0){ u32 a[NM],b[NM]; memcpy(y,y0,sizeof(a));
  for(int it=0;it<NEWTONV;it++){ v_mul(a,t,y); v_scale(a,a); v_sub(b,C_2S,a); v_mul(a,y,b); v_scale(y,a); } }
// the row tail: SUM (ring sum of the terms, one residue per ring, mod each ring) -> ln(sum of exp(l - mx)) as a double (read-out / S)
RH_NOINL double rh_lnsum(const u32*SUMR){
  u32 sum[NM],t[NM],tt[NM],y0[NM],v[NM],U[NM],U2[NM],acc[NM],lnt[NM],lse[NM],tp1[NM];
  for(int i=0;i<NM;i++) sum[i]=SUMR[i]%RH_MOD[i];
  v_scale(sum,sum); v_scale(sum,sum);
  int K=0; for(int j=1;j<=KMAXV;j++){ v_add(t,sum,C_NSJ[j]); K+=v_nonneg(t); }
  v_mul(t,sum,C_T2[K+1]); v_scale(t,t); v_scale(tt,t);
  v_mul(t,tt,C_32); v_scale(t,t); v_sub(y0,C_48,t); u32 y[NM]; rh_newton(y,tt,y0);
  v_add(tp1,tt,C_S); rh_newton(v,tp1,C_175);
  v_add(t,tt,C_NS); v_mul(t,t,v); v_scale(U,t); v_mul(t,U,U); v_scale(U2,t);
  memcpy(acc,C_AT[ATJ],sizeof(acc));
  for(int j=ATJ-1;j>=0;j--){ v_mul(acc,acc,U2); v_scale(acc,acc); v_add(acc,acc,C_AT[j]); }
  v_mul(t,acc,U); v_scale(t,t); v_mulk(lnt,t,2);
  v_mulk(t,C_LN2,(u32)(K+1)); v_add(lse,t,lnt);
  return (double)v_read(lse) / SVAL;
}
"""
OLDTERM = '  v_mul(q,w,C_INVLN2); v_scale(q,q); v_scale(z,q);\n  v_mul(tt,z,C_LN2_2); v_scale(tt,tt); v_sub(rr,w,tt); v_neg(nr,rr);\n  memcpy(acc,C_EXP[EXPD],sizeof(acc));\n  for(int k=EXPD-1;k>=0;k--){ v_mul(acc,acc,nr); v_scale(acc,acc); v_add(acc,acc,C_EXP[k]); }\n'
FASTTERM = '  v_sc(q,w,C_INVLN2,0); v_scale(z,q); v_sc(tt,z,C_LN2_2,0);\n  v_sub(nr,tt,w);\n  memcpy(acc,C_EXP[EXPD],sizeof(acc));\n  #pragma unroll 1\n  for(int k=EXPD-1;k>=0;k--) v_sc(acc,acc,nr,C_EXP[k]);\n'
if FAST: assert OLDTERM in tail; tail = tail.replace(OLDTERM, FASTTERM)
tail = (tail.replace("T2GTAB", T2G).replace("MODTAB", mod_tab).replace("SVAL", f"{S}.0").replace("EXPD", str(EXP_D)).replace("ZMAXV", str(ZMAX))
        .replace("NEWTONV", str(NEWTON)).replace("KMAXV", str(KMAX)).replace("ATJ", str(ATANH_J)))
hdr = ("// ring_head.h — emitted by armilla-astrolabe experiments/abacus_nl/ring_head_emit.py (ABACUS-NL-1 step 3); do not edit: re-emit.\n"
       "// ring softmax / log-sum-exp primitives over the 20 outer T50 moduli, host + device from one source (needs OM_HD, u32/i64 from the trainer header)\n"
       "typedef uint32_t u32; typedef uint64_t u64; typedef int64_t i64;\n#define DEV OM_HD\n#define NM 20\n#define NRR 15\n")
INTTERM = '\n// ABACUS-NL-2 step 1b: the SAME element term from the integer twin (ring_nl IntOps: SCALE = floor division by S, bit-equal to the rings, T1), in 64-bit integer\n// arithmetic, then projected into the rings by v_set_i64: d -> z, r -> Horner on the integers -> residues of the mantissa -> x PULL T2[z].\n// Bit-identical to rh_term for 0 <= d < 2^47 (anything else takes the ring path).\n#ifdef __CUDA_ARCH__\n#define RH_UMULHI64(a,b) __umul64hi((a),(b))\n#else\n#define RH_UMULHI64(a,b) ((u64)(((unsigned __int128)(a)*(b))>>64))\n#endif\n#define RH_S SVAL_U64\n#define RH_MF MF_U64\nstatic const u64 H_C_EXP_I[EXPD1] = CEXPI;\n#ifdef OM_CUDA\n__constant__ u64 D_C_EXP_I[EXPD1] = CEXPI;\n#endif\n#if defined(__CUDA_ARCH__)\n#define C_EXP_I D_C_EXP_I\n#else\n#define C_EXP_I H_C_EXP_I\n#endif\nOM_HD u64 rh_divS(u64 A, u64 *r) { u64 q = RH_UMULHI64(A, RH_MF); u64 rr = A - q * RH_S; if (rr >= RH_S) { q++; rr -= RH_S; } *r = rr; return q; }\n// floor(a*b/S) for |a| < 2^41, |b| < 2^40\nOM_HD i64 rh_mulS(i64 a, i64 b) {\n  const bool ng = (a < 0) != (b < 0); const u64 ua = a < 0 ? (u64)0 - (u64)a : (u64)a, ub = b < 0 ? (u64)0 - (u64)b : (u64)b;\n  u64 ra, rb; const u64 qa = rh_divS(ua * (ub >> 20), &ra); const u64 qb = rh_divS((ra << 20) + ua * (ub & 0xFFFFFu), &rb);\n  const u64 q = (qa << 20) + qb; return ng ? -(i64)q - (rb != 0) : (i64)q;\n}\nRH_NOINL int rh_term_int(u32*t, i64 d) {\n  if (d < 0 || d >= ((i64)1 << 47)) return rh_term(t, d);\n  const u64 dh = (u64)d >> 23, dl = (u64)d & 0x7FFFFFu; u64 ra, rb, rz;\n  const u64 qa = rh_divS(dh * CINV_U64, &ra); const u64 qb = rh_divS((ra << 23) + dl * CINV_U64, &rb); const u64 Q = (qa << 23) + qb;\n  const u64 z = rh_divS(Q, &rz); if (z >= 39203u) return -1;\n  u64 rt; const u64 tt = z * LN22A_U64 + rh_divS(z * LN22B_U64, &rt); const i64 R = d - (i64)tt;\n  if (R < 0 || R >= ((i64)1 << 40)) return rh_term(t, d);\n  i64 acc = (i64)C_EXP_I[EXPD]; for (int k = EXPD - 1; k >= 0; k--) acc = rh_mulS(acc, -R) + (i64)C_EXP_I[k];\n  u32 ar[NM]; v_set_i64(ar, acc); v_mul(t, ar, RH_T2ROW(z > ZMAXV ? ZMAXV : (int)z)); return (int)z;\n}\n'

# ABACUS-NL-2 lever 2: the mantissa Horner in a SUB-BASE of 11 rings (6 base + the 5 scale rings); the front end (z, r) stays on all 20 rings, the mantissa is
# extended back to 20 rings exactly (Garner digits of the 6 base rings; acc < 2^41 < their product 2^45.1). Range proof: results/ringhead/nl2/SUBBASE.md.
Rb = J.RR[:J.SUB_NB]; Ms = Rb + J.G; PB = [1]
for m in Rb: PB.append(PB[-1] * m)
Rd = [[(-(PB[j] % Rb[i]) * pow(PB[i], -1, Rb[i])) % Rb[i] for j in range(i)] for i in range(len(Rb))]
ext = ["u32 dg[%d];" % len(Rb)]
for i, m in enumerate(Rb):
    dot = "".join(f"+dg[{j}]*{Rd[i][j]}u" for j in range(i))
    ext.append(f"dg[{i}]=(acc[{i}]*{pow(PB[i], -1, m)}u{dot})%{m}u;")
for i in range(J.NRR):
    if i < len(Rb): ext.append(f"ar[{i}]=acc[{i}];")
    else:
        m = J.MODS[i]; cs = [PB[j] % m for j in range(len(Rb))]
        ext.append(f"ar[{i}]=(" + "+".join(f"dg[{j}]*{cs[j]}u" for j in range(len(Rb))) + f")%{m}u;")
for gi in range(5): ext.append(f"ar[{J.NRR + gi}]=acc[{len(Rb) + gi}];")
SUBTERM2 = r"""
RH_NOINL int rh_term_sub2(u32*t, i64 d) {                 // front end on 12 rings (d < 2^47), Horner on 11, extension + PULL on 20
  if (d < 0 || d >= ((i64)1 << 47)) return rh_term_sub(t, d);
  u32 w[NMF],q[NMF],z[NMF],tt[NMF],nr[NMF],ns[NMS],acc[NMS],ar[NM];
  v_setf(w,d); v_scf(q,w,C_INVLN2F,0); v_scf(z,q,(const u32*)0,0); v_scf(tt,z,C_LN2_2F,0); v_subf(nr,tt,w);
  const i64 zi=v_small(z); if(zi<0) return -1;
  NSCOPY2
  for(int i=0;i<NMS;i++) acc[i]=C_EXPX[EXPD][i];
  #pragma unroll 1
  for(int k=EXPD-1;k>=0;k--) v_scs(acc,acc,ns,C_EXPX[k]);
  EXTEND
  v_mul(t,ar,RH_T2ROW(zi>ZMAXV?ZMAXV:(int)zi)); return (int)zi;
}
"""
SUBTERM2 = SUBTERM2.replace("NSCOPY2", " ".join([f"ns[{i}]=nr[{i}];" for i in range(len(Rb))] + [f"ns[{len(Rb) + g}]=nr[{len(Rb) + 1 + g}];" for g in range(5)])).replace("EXTEND", "\n  ".join(ext))
SUBTERM = r"""
RH_NOINL int rh_term_sub(u32*t, i64 d) {
  u32 w[NM],q[NM],z[NM],tt[NM],nr[NM],ns[NMS],acc[NMS],ar[NM];
  if (d < 0) return rh_term(t, d);                      // outside the head's contract (d = qmx - q >= 0): the full-ring path
  v_set_i64(w,d); v_sc(q,w,C_INVLN2,0); v_scale(z,q); v_sc(tt,z,C_LN2_2,0); v_sub(nr,tt,w);
  const i64 zi=v_small(z); if(zi<0) return -1;
  NSCOPY
  for(int i=0;i<NMS;i++) acc[i]=C_EXPX[EXPD][i];
  #pragma unroll 1
  for(int k=EXPD-1;k>=0;k--) v_scs(acc,acc,ns,C_EXPX[k]);
  EXTEND
  v_mul(t,ar,RH_T2ROW(zi>ZMAXV?ZMAXV:(int)zi)); return (int)zi;
}
"""
SUBTERM_X = 0
SUBTERM = SUBTERM.replace("NSCOPY", " ".join([f"ns[{i}]=nr[{i}];" for i in range(len(Rb))] + [f"ns[{len(Rb) + g}]=nr[{J.NRR + g}];" for g in range(5)])).replace("EXTEND", "\n  ".join(ext))

cexpi = "{" + ", ".join(f"{c}ull" for c in RN.C_EXP) + "}"
INTTERM = (INTTERM.replace("SVAL_U64", f"{S}ull").replace("MF_U64", f"{(1 << 64) // S}ull").replace("CEXPI", cexpi).replace("EXPD1", str(EXP_D + 1)).replace("EXPD", str(EXP_D))
           .replace("CINV_U64", f"{RN.C_INVLN2}ull").replace("LN22A_U64", f"{RN.C_LN2_2 // S}ull").replace("LN22B_U64", f"{RN.C_LN2_2 % S}ull").replace("ZMAXV", str(ZMAX)))
tail = tail.replace("@@INTTERM@@", (INTTERM + (SUBTERM + SUBTERM2).replace("EXPD", str(EXP_D)).replace("ZMAXV", str(ZMAX))) if FAST else "")
open(sys.argv[1], "w").write(hdr + "\n".join(lines) + tail)
print("wrote", sys.argv[1])
