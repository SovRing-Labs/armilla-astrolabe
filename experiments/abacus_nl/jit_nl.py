#!/usr/bin/env python3
"""
Armilla — the JIT centre renders the ring non-linear math (ABACUS-NL-0 step 2, 2026-10-13).

The algorithm of ring_nl.py (exp, 1/x, ln, softmax, log-sum-exp, cross-entropy, all as ring operations) is rendered as C
with every constant baked in: the 20 outer moduli, MRC inverses, the scale-group CRT, S^-1 per ring, the bias, the exp /
atanh / Newton coefficients and the PULL table T[z] = round(S^2 2^-z), each as residues. Every per-ring line is unrolled
with its modulus as a literal, so the compiler turns each "mod m" into a multiply (the G9 JIT-centre method: specialise
the kernel to constants known at run time). gcc at run time, loaded by ctypes, fired one row at a time.
Input: float64 logits (rint(x * S) on entry, as ring_nl.quantise). Output: softmax mantissa P (p = P / S * 2^-E), exponent
E, log-sum-exp and loss as fixed-point integers (value = X / S).
Pass rules fixed before the run:
  J1 bit-identity   kernel == ring_nl Algo (IntOps, proven == RingOps in step 1) on P, E, LSE, LOSS     PASS = 0 mismatches
  J2 speed          microseconds per row (n = 512) and per element, one core; float64 softmax in C as the reference cost
Writes jit_nl.result.json.
"""
import ctypes, json, math, os, random, subprocess, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ring_nl as RN
from ring_nl import MODS, RR, G, S, B, M, NEWTON, EXP_D, ATANH_J, ZMAX, KMAX, BIAS_K

NM, NRR = len(MODS), len(RR)
INV = [[pow(MODS[j], -1, MODS[i]) if j < i else 0 for j in range(NM)] for i in range(NM)]

def res(c): return [c % m for m in MODS]
def carr(name, vals): return f"CONST u32 {name}[{len(vals)}] = {{{', '.join(map(str, vals))}}};"

def render(cuda=False):
    return render_prims(cuda) + ("" if cuda else CPU_ROW)


SUB_NB = 6; SUB_BIAS = 42; FRONT_BIAS = 48          # sub-base: 6 base rings + the 5 scale rings; bias B' = S 2^42 (see SUBBASE.md for the range proof)
def res_in(c, mods): return [c % m for m in mods]
def vsc_text(Rb, fname, NB, cb, cbs, bexp):
    """C text of v_sc over the ring list Rb + G (the base rings first, the scale group last): z = SCALE(a*b) + e, exact (bias B = S 2^bexp)."""
    nb = len(Rb); Ms = Rb + G
    P = [1]
    for m in Rb: P.append(P[-1] * m)
    Mb = P[nb]; assert Mb > 1.6 * (1 << bexp) * 4, "q = 2^bexp (+-) must stay below the base product with margin"
    pk = lambda v: sum((int(x) & 255) << (8 * j) for j, x in enumerate(v))
    bd = [f"u32 w[{NB}], tg[5];"]
    for gi, g in enumerate(G):
        i = nb + gi
        bd.append(f"tg[{gi}]=(((b?a[{i}]*b[{i}]:a[{i}])+{cb}[{i}])*{pow(S // g, -1, g) % g}u)%{g}u;")
    bd.append("u64 sg=(" + " + ".join(f"(u64)tg[{gi}]*{S // g}ull" for gi, g in enumerate(G)) + f"); u32 k=(sg>={S}ull)+(sg>={2 * S}ull)+(sg>={3 * S}ull)+(sg>={4 * S}ull);")
    bd.append("u32 W0=tg[0]|(tg[1]<<8)|(tg[2]<<16)|(tg[3]<<24), W1=tg[4]|(k<<8);")
    for i in range(nb):
        m = Ms[i]
        c0 = pk([(S // g) % m for g in G[:4]]); c1 = pk([(S // G[4]) % m, m - S % m])
        big = m * (-(-(5 * 233 * 233 + 4 * 233) // m))
        bd.append(f"w[{i}]=(b?a[{i}]*b[{i}]:a[{i}])+{cb}[{i}]+{big}u-DP4A(W0,{c0}u,DP4A(W1,{c1}u,0u));")
    bd.append(f"u32 xi[{nb}];")
    for i in range(nb):
        m = Ms[i]; c = pow(S, -1, m) * pow((Mb // m) % m, -1, m) % m
        bd.append(f"xi[{i}]=(w[{i}]*{c}u)%{m}u;")
    bd.append("u64 al=" + "+".join(f"(u64)xi[{i}]*{-(-(1 << 48) // Ms[i])}ull" for i in range(nb)) + "; u32 al32=(u32)(al>>48);")
    nw = -(-nb // 4)
    bd.append("u32 " + ", ".join(f"X{q}=" + "|".join(f"(xi[{j}]<<{8 * (j - 4 * q)})" if j > 4 * q else f"xi[{j}]" for j in range(4 * q, min(4 * q + 4, nb))) for q in range(nw)) + ";")
    for i in range(nb):
        m = Ms[i]; iS = pow(S, -1, m)
        bd.append(f"z[{i}]=(w[{i}]*{iS}u+{m}u-{cbs}[{i}]+(e?e[{i}]:0u))%{m}u;")
    for gi, g in enumerate(G):
        i = nb + gi
        kw = [pk([(Mb // Ms[j]) % g for j in range(q, min(q + 4, nb))]) for q in range(0, nb, 4)]
        dot = "0u"
        for q in reversed(range(nw)): dot = f"DP4A(X{q},{kw[q]}u,{dot})"
        bd.append(f"z[{i}]=({dot}+al32*{g - Mb % g}u+{g}u-{cbs}[{i}]+(e?e[{i}]:0u))%{g}u;")
    return f"DEV void {fname}(u32*z,const u32*a,const u32*b,const u32*e) {{ " + " ".join(bd) + " }"

def render_prims(cuda=False, fast=False, sub=False):
    L = []; A = L.append
    A("#include <stdint.h>\n#include <math.h>\n#include <string.h>")
    A("#define CONST __constant__\n#define DEV static __device__ __forceinline__" if cuda else "#define CONST static const\n#define DEV static inline")
    A("typedef uint32_t u32; typedef uint64_t u64; typedef int64_t i64;")
    A(f"#define NM {NM}\n#define NRR {NRR}")
    consts = {"C_B": res(B), "C_BS": res(B // S), "C_INVLN2": res(RN.C_INVLN2), "C_LN2": res(RN.C_LN2),
              "C_LN2_2": res(RN.C_LN2_2), "C_48": res(RN.C48), "C_32": res(RN.C32), "C_2S": res(2 * S),
              "C_S": res(S), "C_NS": res(-S), "C_175": res(RN.C_1_175)}
    for k, v in consts.items(): A(carr(k, v))
    A(f"CONST u32 C_EXP[{EXP_D + 1}][NM] = {{{', '.join('{' + ', '.join(map(str, res(c))) + '}' for c in RN.C_EXP)}}};")
    A(f"CONST u32 C_AT[{ATANH_J + 1}][NM] = {{{', '.join('{' + ', '.join(map(str, res(c))) + '}' for c in RN.C_ATANH)}}};")
    A(f"CONST u32 C_T2[{ZMAX + 1}][NM] = {{{', '.join('{' + ', '.join(map(str, res(c))) + '}' for c in RN.T2)}}};")
    A(f"CONST u32 C_NSJ[{KMAX + 1}][NM] = {{{', '.join('{' + ', '.join(map(str, res(-(S << j)))) + '}' for j in range(KMAX + 1))}}};")
    bd = RN.RingOps().B_digits
    A(carr("B_DIG", [int(d) for d in bd]))
    # element-wise ring ops, unrolled with literal moduli
    def fn(name, args, body): A(f"DEV void {name}({args}) {{ " + " ".join(body) + " }")
    fn("v_add", "u32*z,const u32*a,const u32*b", [f"z[{i}]=(a[{i}]+b[{i}])%{m}u;" for i, m in enumerate(MODS)])
    fn("v_sub", "u32*z,const u32*a,const u32*b", [f"z[{i}]=(a[{i}]+{m}u-b[{i}])%{m}u;" for i, m in enumerate(MODS)])
    fn("v_mul", "u32*z,const u32*a,const u32*b", [f"z[{i}]=(a[{i}]*b[{i}])%{m}u;" for i, m in enumerate(MODS)])
    fn("v_neg", "u32*z,const u32*a", [f"z[{i}]=({m}u-a[{i}])%{m}u;" for i, m in enumerate(MODS)])
    fn("v_mulk", "u32*z,const u32*a,u32 k", [f"z[{i}]=(a[{i}]*(k%{m}u))%{m}u;" for i, m in enumerate(MODS)])
    # mixed-radix digits over the first k rings
    def mrc_lines(src, k, dst="d"):
        out = [f"u32 {dst}[{k}];"]
        for i in range(k):
            mi = MODS[i]; out.append(f"{{ u32 q_={src}[{i}];")
            for j in range(i):
                out.append(f"q_=((q_+{mi}u-({dst}[{j}]%{mi}u))*{INV[i][j]}u)%{mi}u;")
            out.append(f"{dst}[{i}]=q_; }}")
        return out
    # SCALE: exact floor(X / S)
    body = [f"u32 x[NM];"] + [f"x[{i}]=(a[{i}]+C_B[{i}])%{m}u;" for i, m in enumerate(MODS)]
    terms = [f"(u64)((x[{NRR + gi}]*{pow(S // g, -1, g)}u)%{g}u)*{S // g}ull" for gi, g in enumerate(G)]
    body.append(f"u64 am=({' + '.join(terms)})%{S}ull;")
    body.append("u32 y[NM];")
    body += [f"y[{i}]=((x[{i}]+{m}u-(u32)(am%{m}ull))%{m}u)*{pow(S, -1, m)}u%{m}u;" for i, m in enumerate(RR)]
    body += mrc_lines("y", NRR)
    for gi, g in enumerate(G):
        ex = f"d[{NRR - 1}]%{g}u"
        body.append(f"{{ u32 v={ex};")
        for i in range(NRR - 2, -1, -1):
            body.append(f"v=(v*{RR[i] % g}u+d[{i}])%{g}u;")
        body.append(f"y[{NRR + gi}]=v; }}")
    body += [f"z[{i}]=(y[{i}]+{m}u-C_BS[{i}])%{m}u;" for i, m in enumerate(MODS)]
    if not fast: fn("v_scale", "u32*z,const u32*a", body)
    else:
        A("#if defined(__CUDA_ARCH__)\n#define DP4A(a,b,c) __dp4a((unsigned)(a),(unsigned)(b),(unsigned)(c))\n#else\n"
          "static inline u32 DP4A(u32 a,u32 b,u32 c){ return c+(a&255u)*(b&255u)+((a>>8)&255u)*((b>>8)&255u)+((a>>16)&255u)*((b>>16)&255u)+(a>>24)*(b>>24); }\n#endif")
        A(vsc_text(RR, "v_sc", "NRR", "C_B", "C_BS", BIAS_K))
        A("DEV void v_scale(u32*z,const u32*a) { v_sc(z,a,(const u32*)0,(const u32*)0); }")
        if sub:     # ABACUS-NL-2 lever 2: the SAME exact SCALE on a sub-base (6 base rings + the 5 scale rings = 11) for the Horner steps; see SUBBASE.md
            Rb = RR[:SUB_NB]; Ms = Rb_mods = Rb + G
            A(f"#define NMS {len(Ms)}\n#define NBS {SUB_NB}")
            A(carr("C_BX", res_in(S << SUB_BIAS, Ms)).replace("CONST u32 C_BX[", "CONST u32 C_BX[")); A(carr("C_BSX", res_in(1 << SUB_BIAS, Ms)))
            A(vsc_text(Rb, "v_scs", "NBS", "C_BX", "C_BSX", SUB_BIAS))
            FNB = SUB_NB + 1; Mf = RR[:FNB] + G       # the front end (d -> z, r) for d < 2^47 on 7 base rings + the 5 scale rings = 12, bias S 2^48
            A(f"#define NMF {len(Mf)}\n#define NBF {FNB}")
            A(carr("C_BF", res_in(S << FRONT_BIAS, Mf))); A(carr("C_BSF", res_in(1 << FRONT_BIAS, Mf)))
            A(carr("C_INVLN2F", res_in(RN.C_INVLN2, Mf))); A(carr("C_LN2_2F", res_in(RN.C_LN2_2, Mf)))
            A(vsc_text(RR[:FNB], "v_scf", "NBF", "C_BF", "C_BSF", FRONT_BIAS))
            A("DEV void v_setf(u32*z,i64 v){ int ng=v<0; u64 u=ng?(u64)0-(u64)v:(u64)v; u32 lo=(u32)u, hi=(u32)(u>>32); " + " ".join(
                f"{{ u32 r=((hi%{m}u)*{(1 << 32) % m}u+lo%{m}u)%{m}u; z[{i}]=(ng&&r)?{m}u-r:r; }}" for i, m in enumerate(Mf)) + " }")
            A("DEV void v_subf(u32*z,const u32*a,const u32*b){ " + " ".join(f"z[{i}]=(a[{i}]+{m}u-b[{i}])%{m}u;" for i, m in enumerate(Mf)) + " }")
            A(f"CONST u32 C_EXPX[{EXP_D + 1}][NMS] = {{{', '.join('{' + ', '.join(map(str, res_in(c, Ms))) + '}' for c in RN.C_EXP)}}};")
    # NONNEG: X >= 0 by comparing the mixed-radix digits of X + B with those of B
    body = ["u32 x[NM];"] + [f"x[{i}]=(a[{i}]+C_B[{i}])%{m}u;" for i, m in enumerate(MODS)] + mrc_lines("x", NM)
    body.append("for(int i=NM-1;i>=0;i--){ if(d[i]>B_DIG[i]) return 1; if(d[i]<B_DIG[i]) return 0; } return 1;")
    A("DEV int v_nonneg(const u32*a) { " + " ".join(body) + " }")
    # small integer read-out (two digits), -1 if too large
    body = mrc_lines("a", 3) + [f"if(d[2]) return -1; return (i64)d[0]+(i64)d[1]*{MODS[0]};"]
    A("DEV i64 v_small(const u32*a) { " + " ".join(body) + " }")
    # signed read-out of |X| < 2^62 (mixed radix weights mod 2^64)
    w = []; acc = 1
    for m in MODS: w.append(acc % (1 << 64)); acc *= m
    body = ["u32 x[NM];"] + [f"x[{i}]=(a[{i}]+C_B[{i}])%{m}u;" for i, m in enumerate(MODS)] + mrc_lines("x", NM)
    body.append("u64 v=0;" + "".join(f"v+=(u64)d[{i}]*{w[i]}ull;" for i in range(NM)) + f"v-={B % (1 << 64)}ull; return (i64)v;")
    A("DEV i64 v_read(const u32*a) { " + " ".join(body) + " }")
    if not fast:
        A("DEV void v_set_i64(u32*z,i64 v){ " + " ".join(
            f"{{ i64 r=v%{m}; z[{i}]=(u32)(r<0?r+{m}:r); }}" for i, m in enumerate(MODS)) + " }")
    else:   # |v| split in 32-bit halves: 32-bit reductions instead of a signed 64-bit remainder per ring (same residues)
        A("DEV void v_set_i64(u32*z,i64 v){ int ng=v<0; u64 u=ng?(u64)0-(u64)v:(u64)v; u32 lo=(u32)u, hi=(u32)(u>>32); " + " ".join(
            f"{{ u32 r=((hi%{m}u)*{(1 << 32) % m}u+lo%{m}u)%{m}u; z[{i}]=(ng&&r)?{m}u-r:r; }}" for i, m in enumerate(MODS)) + " }")
    return "\n".join(L)

CPU_ROW = """
static inline void v_scale2(u32*z,const u32*a){ u32 t[NM]; v_scale(t,a); v_scale(z,t); }
static inline void newton(u32*y,const u32*t,const u32*y0){ u32 a[NM],b[NM]; memcpy(y,y0,sizeof(a));
  for(int it=0;it<%(NEWTON)d;it++){ v_mul(a,t,y); v_scale(a,a); v_sub(b,C_2S,a); v_mul(a,y,b); v_scale(y,a); } }
/* one row: x[n] float64 logits; out P[n], E[n], *LSE, *LOSS (tgt<0: no loss). Returns 0, or -1 on a range error. */
int row_core(int n,int tgt,i64*P,int*E,i64*LSE,i64*LOSS,u32*W /* n*NM residues, overwritten */,u32*MN /* n*NM */){
  u32 mx[NM],t[NM];
  memcpy(mx,W,sizeof(mx));
  for(int i=1;i<n;i++){ v_sub(t,W+i*NM,mx); if(v_nonneg(t)) memcpy(mx,W+i*NM,sizeof(mx)); }
  u32 xt[NM]; if(tgt>=0) memcpy(xt,W+tgt*NM,sizeof(xt));
  u32 sum[NM]; memset(sum,0,sizeof(sum));
  for(int i=0;i<n;i++){
    u32 *w=W+i*NM, q[NM], z[NM], r[NM], nr[NM], acc[NM];
    v_sub(w,mx,w);                                   /* w = max - x >= 0 */
    v_mul(q,w,C_INVLN2); v_scale(q,q); v_scale(z,q); /* z = floor(w / ln2) */
    v_mul(t,z,C_LN2_2); v_scale(t,t); v_sub(r,w,t); v_neg(nr,r);
    memcpy(acc,C_EXP[%(EXP_D)d],sizeof(acc));
    for(int k=%(EXP_D)d-1;k>=0;k--){ v_mul(acc,acc,nr); v_scale(acc,acc); v_add(acc,acc,C_EXP[k]); }
    i64 zi=v_small(z); if(zi<0) return -1; E[i]=(int)zi;
    memcpy(MN+i*NM,acc,sizeof(acc));
    v_mul(t,acc,C_T2[zi>%(ZMAX)d?%(ZMAX)d:zi]); v_add(sum,sum,t);   /* PULL T[z], lazy (unscaled) sum */
  }
  v_scale2(sum,sum);
  int K=0; for(int j=1;j<=%(KMAX)d;j++){ v_add(t,sum,C_NSJ[j]); K+=v_nonneg(t); }
  u32 tt[NM],y0[NM],y[NM]; v_mul(t,sum,C_T2[K+1]); v_scale2(tt,t);
  v_mul(t,tt,C_32); v_scale(t,t); v_sub(y0,C_48,t); newton(y,tt,y0);
  for(int i=0;i<n;i++){ v_mul(t,MN+i*NM,y); v_scale(t,t); P[i]=v_read(t); E[i]+=K+1; }
  u32 tp1[NM],v[NM],U[NM],U2[NM],acc[NM],lnt[NM],lse[NM];
  v_add(tp1,tt,C_S); newton(v,tp1,C_175);
  v_add(t,tt,C_NS); v_mul(t,t,v); v_scale(U,t); v_mul(t,U,U); v_scale(U2,t);
  memcpy(acc,C_AT[%(ATANH_J)d],sizeof(acc));
  for(int j=%(ATANH_J)d-1;j>=0;j--){ v_mul(acc,acc,U2); v_scale(acc,acc); v_add(acc,acc,C_AT[j]); }
  v_mul(t,acc,U); v_scale(t,t); v_mulk(lnt,t,2);
  v_mulk(t,C_LN2,(u32)(K+1)); v_add(lse,mx,t); v_add(lse,lse,lnt);
  *LSE=v_read(lse);
  if(tgt>=0){ v_sub(t,lse,xt); *LOSS=v_read(t); }
  return 0;
}
int row_softmax_lse(const double*x,int n,int tgt,i64*P,int*E,i64*LSE,i64*LOSS,u32*W,u32*MN){
  double sd=(double)%(S)dull; for(int i=0;i<n;i++) v_set_i64(W+i*NM,(i64)rint(x[i]*sd));
  return row_core(n,tgt,P,E,LSE,LOSS,W,MN);
}
int batch_res(const u32*X,int rows,int n,const int*tgt,i64*P,int*E,i64*LSE,i64*LOSS,u32*W,u32*MN){
  for(int r=0;r<rows;r++){ memcpy(W,X+(i64)r*n*NM,(size_t)n*NM*sizeof(u32));
    if(row_core(n,tgt?tgt[r]:-1,P+(i64)r*n,E+(i64)r*n,LSE+r,LOSS+r,W,MN)) return -1-r; }
  return 0;
}
int batch(const double*x,int rows,int n,const int*tgt,i64*P,int*E,i64*LSE,i64*LOSS,u32*W,u32*MN){
  for(int r=0;r<rows;r++) if(row_softmax_lse(x+(i64)r*n,n,tgt?tgt[r]:-1,P+(i64)r*n,E+(i64)r*n,LSE+r,LOSS+r,W,MN)) return -1-r;
  return 0;
}
/* float64 reference cost: plain stable softmax + LSE */
void f64_softmax(const double*x,int rows,int n,double*p,double*lse){
  for(int r=0;r<rows;r++){ const double*a=x+(i64)r*n; double m=a[0]; for(int i=1;i<n;i++) if(a[i]>m) m=a[i];
    double s=0; for(int i=0;i<n;i++){ p[(i64)r*n+i]=exp(a[i]-m); s+=p[(i64)r*n+i]; }
    for(int i=0;i<n;i++) p[(i64)r*n+i]/=s; lse[r]=m+log(s); } }
""" % {"NEWTON": NEWTON, "S": S, "EXP_D": EXP_D, "ZMAX": ZMAX, "KMAX": KMAX, "ATANH_J": ATANH_J}

def build(src, tag="nl"):
    tmp = tempfile.mkdtemp(prefix="armilla-jit-"); c = os.path.join(tmp, f"{tag}.c"); so = os.path.join(tmp, f"{tag}.so")
    open(c, "w").write(src)
    t0 = time.perf_counter()
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, c, "-lm"], check=True)
    return ctypes.CDLL(so), time.perf_counter() - t0, c

class Kernel:
    def __init__(self, src=None):
        self.src = src or render(); self.lib, self.compile_s, self.c_path = build(self.src)
    def run(self, x, tgt=None):
        x = np.ascontiguousarray(x, dtype=np.float64); rows, n = x.shape
        P = np.zeros((rows, n), np.int64); E = np.zeros((rows, n), np.int32)
        LSE = np.zeros(rows, np.int64); LOSS = np.zeros(rows, np.int64)
        W = np.zeros(n * NM, np.uint32); MN = np.zeros(n * NM, np.uint32)
        vp = ctypes.c_void_p; ptr = lambda a: a.ctypes.data_as(vp)
        t = None if tgt is None else np.ascontiguousarray(tgt, dtype=np.int32)
        rc = self.lib.batch(ptr(x), rows, n, ptr(t) if t is not None else None, ptr(P), ptr(E), ptr(LSE), ptr(LOSS), ptr(W), ptr(MN))
        if rc: raise RuntimeError(f"ring range error at row {-1 - rc}")
        return P, E, LSE, LOSS

def compare(k, rows, n, seed, spread):
    rnd = random.Random(seed)
    xs = np.array([[rnd.gauss(0, spread) for _ in range(n)] for _ in range(rows)])
    tgt = np.array([rnd.randrange(n) for _ in range(rows)])
    P, E, LSE, LOSS = k.run(xs, tgt)
    ref = RN.Algo(RN.IntOps()).softmax_lse(RN.IntOps().lift(RN.quantise(xs.tolist())), tgt)
    mism = int((P != ref["P"].astype(np.int64)).sum()) + int((E != np.asarray(ref["E"], np.int64)).sum())
    mism += int((LSE != ref["LSE"][:, 0].astype(np.int64)).sum()) + int((LOSS != ref["LOSS"][:, 0].astype(np.int64)).sum())
    return mism

if __name__ == "__main__":
    k = Kernel()
    R = {"date": "2026-10-13", "compile_s": round(k.compile_s, 3), "c_lines": k.src.count("\n") + 1,
         "gcc": subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]}
    R["J1"] = {f"rows{r}_n{n}_spread{s}": compare(k, r, n, sd, s)
               for (r, n, sd, s) in [(64, 512, 0, 8.0), (64, 512, 1, 30.0), (200, 40, 2, 25.0), (16, 4096, 3, 4.0)]}
    rnd = np.random.default_rng(9); xs = rnd.normal(0, 8, (256, 512))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); k.run(xs); ts.append(time.perf_counter() - t0)
    tr = sorted(ts)[2]
    p = np.zeros_like(xs); l = np.zeros(256); vp = ctypes.c_void_p
    tf = []
    for _ in range(5):
        t0 = time.perf_counter(); k.lib.f64_softmax(xs.ctypes.data_as(vp), 256, 512, p.ctypes.data_as(vp), l.ctypes.data_as(vp)); tf.append(time.perf_counter() - t0)
    tf = sorted(tf)[2]
    R["J2"] = {"rows": 256, "n": 512, "ring_us_per_row": round(tr / 256 * 1e6, 1), "ring_ns_per_element": round(tr / 256 / 512 * 1e9, 1),
               "f64_us_per_row": round(tf / 256 * 1e6, 2), "ring_over_f64": round(tr / tf, 1)}
    R["verdict"] = "PASS" if all(v == 0 for v in R["J1"].values()) else "FAIL"
    json.dump(R, open(os.path.join(HERE, "jit_nl.result.json"), "w"), indent=1); print(json.dumps(R, indent=1))
