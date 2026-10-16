#!/usr/bin/env python3
"""
Armilla — the ring GEMM and the ring softmax / log-sum-exp / loss on a GPU, against the CPU JIT kernel
(ABACUS-NL-0 step 4, 2026-10-13).

One renderer, two back-ends: jit_nl.render_prims(cuda=True) emits the SAME ring primitives (unrolled, literal moduli,
constants baked) as __device__ functions and __constant__ tables. The step, start to end inside the rings:
  logits    X = (A . W^T) * C_s        A int8 activations (rows x K), W ternary (V x K), C_s = round(S / (127 sqrt K)):
                                       a ternary GEMM is adds only; done per ring, carry-free (k_gemm)
  max       exact ring COMPARE scan per row (k_max)
  exp       per element: mantissa + exponent, term = mantissa * PULL T[z] (k_exp); the row sum is atomicAdd on each ring:
            the GPU adds in whatever order its threads arrive, and the result cannot change (rings are order-free)
  tail      per row: SCALE, K, Newton 1/t, ln t, LSE, loss (k_row);  softmax mantissa per element (k_p)
CPU twin: the same GEMM per ring in NumPy (int64, exact), then jit_nl batch_res (the CPU JIT kernel).
Pass rules fixed before the run:
  G1 bit-identity   GPU P, E, LSE, LOSS == CPU JIT kernel, every entry, on 2 shapes        PASS = 0 mismatches
  G2 order          the GPU run repeated 3 times (atomic order differs run to run)          PASS = identical bits each time
  G3 speed          ms per step on the GPU (kernels only, cudaEvent) vs the CPU kernel      reported
Usage: gpu_nl.py prep DIR  (inputs + CPU twin + .cu)   |   gpu_nl.py check DIR  (after the GPU run wrote out_*.bin)
"""
import ctypes, json, math, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jit_nl as J, ring_nl as RN
from ring_nl import MODS, S, ZMAX, KMAX, EXP_D, ATANH_J, NEWTON
NM = len(MODS)
SHAPES = [("small", 64, 256, 512, 11), ("school", 128, 256, 4096, 12)]   # name, rows, K, V, seed

def cs_of(K): return round(S / (127 * math.sqrt(K)))

def make_inputs(rows, K, V, seed):
    r = np.random.default_rng(seed)
    A = r.integers(-127, 128, (rows, K), dtype=np.int64).astype(np.int8)
    W = r.choice(np.array([-1, 0, 1], np.int8), (V, K), p=[0.3, 0.4, 0.3])
    tgt = r.integers(0, V, rows).astype(np.int32)
    return A, W, tgt

def cpu_twin(A, W, tgt, K, k=None):
    k = k or J.Kernel(); cs = cs_of(K); rows, V = A.shape[0], W.shape[0]
    X = np.empty((rows, V, NM), np.uint32)
    for i, m in enumerate(MODS):                      # ring GEMM on the CPU: per ring, exact int64
        Xi = (A.astype(np.int64) % m) @ (W.astype(np.int64).T % m) % m
        X[:, :, i] = (Xi * (cs % m) % m).astype(np.uint32)
    P = np.zeros((rows, V), np.int64); E = np.zeros((rows, V), np.int32); LSE = np.zeros(rows, np.int64); LOSS = np.zeros(rows, np.int64)
    Wb = np.zeros(V * NM, np.uint32); MN = np.zeros(V * NM, np.uint32); vp = ctypes.c_void_p; p = lambda a: a.ctypes.data_as(vp)
    t0 = time.perf_counter()
    rc = k.lib.batch_res(p(X), rows, V, p(tgt), p(P), p(E), p(LSE), p(LOSS), p(Wb), p(MN))
    dt = time.perf_counter() - t0
    assert rc == 0, rc
    return P, E, LSE, LOSS, dt

def render_cu():
    prims = J.render_prims(cuda=True)
    fin = " ".join(f"o[{i}]=((u32)((s%{m}+{m})%{m})*C_CS[{i}])%{m}u;" for i, m in enumerate(MODS))
    return prims + r"""
#include <stdio.h>
#include <stdlib.h>
__constant__ u32 C_CS[NM];
__global__ void k_gemm(const signed char*A,const signed char*W,int rows,int K,int V,u32*X){
  int r=blockIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=V) return;
  int s=0;                                                       /* exact integer dot product (|s| <= 127*K), then one reduction per ring */
  for(int k=0;k<K;k++){ int w=W[(long)j*K+k]; if(w) s+= w>0 ? (int)A[(long)r*K+k] : -(int)A[(long)r*K+k]; }
  u32*o=X+((long)r*V+j)*NM; FINLINE
}
#define MAXTB 256
__global__ void k_max(const u32*X,int rows,int V,u32*MX){      /* one block per row; max is associative+commutative, so any tree order gives the same ring value */
  __shared__ u32 sm[MAXTB][NM];
  int r=blockIdx.x, tid=threadIdx.x; const u32*x=X+(long)r*V*NM;
  u32 mx[NM],t[NM]; memcpy(mx,x+(long)tid*NM,sizeof(mx));       /* V >= MAXTB */
  for(int i=tid+MAXTB;i<V;i+=MAXTB){ v_sub(t,x+(long)i*NM,mx); if(v_nonneg(t)) memcpy(mx,x+(long)i*NM,sizeof(mx)); }
  memcpy(sm[tid],mx,sizeof(mx)); __syncthreads();
  for(int s=MAXTB/2;s>0;s>>=1){
    if(tid<s){ v_sub(t,sm[tid+s],sm[tid]); if(v_nonneg(t)) memcpy(sm[tid],sm[tid+s],sizeof(mx)); }
    __syncthreads();
  }
  if(tid<NM) MX[(long)r*NM+tid]=sm[0][tid];
}
__global__ void k_exp(const u32*X,int rows,int V,const u32*MX,u32*MN,int*E,u32*SUMR,int*err){
  /* V % blockDim.x == 0: a block never straddles rows. Row sum: shared-memory atomics, then one global atomic per ring per block (ring sums are order-free) */
  __shared__ u32 ss[NM];
  if(threadIdx.x<NM) ss[threadIdx.x]=0; __syncthreads();
  long g=(long)blockIdx.x*blockDim.x+threadIdx.x; int r=(int)(g/V);
  if(g<(long)rows*V){
    u32 w[NM],q[NM],z[NM],t[NM],rr[NM],nr[NM],acc[NM];
    v_sub(w,MX+(long)r*NM,X+g*NM);
    v_mul(q,w,C_INVLN2); v_scale(q,q); v_scale(z,q);
    v_mul(t,z,C_LN2_2); v_scale(t,t); v_sub(rr,w,t); v_neg(nr,rr);
    memcpy(acc,C_EXP[EXPD],sizeof(acc));
    for(int k=EXPD-1;k>=0;k--){ v_mul(acc,acc,nr); v_scale(acc,acc); v_add(acc,acc,C_EXP[k]); }
    i64 zi=v_small(z);
    if(zi<0){ *err=1; } else {
      E[g]=(int)zi;
      memcpy(MN+g*NM,acc,sizeof(acc));
      v_mul(t,acc,C_T2[zi>ZMAXV?ZMAXV:zi]);
      for(int i=0;i<NM;i++) atomicAdd(&ss[i],t[i]);
    }
  }
  __syncthreads();
  if(threadIdx.x<NM && g-threadIdx.x<(long)rows*V) atomicAdd(SUMR+(long)(((long)blockIdx.x*blockDim.x)/V)*NM+threadIdx.x,ss[threadIdx.x]);
}
__device__ void d_newton(u32*y,const u32*t,const u32*y0){ u32 a[NM],b[NM]; memcpy(y,y0,sizeof(a));
  for(int it=0;it<NEWTONV;it++){ v_mul(a,t,y); v_scale(a,a); v_sub(b,C_2S,a); v_mul(a,y,b); v_scale(y,a); } }
__global__ void k_row(const u32*X,int rows,int V,const int*tgt,const u32*MX,const u32*SUMR,u32*Y,int*KR,i64*LSE,i64*LOSS){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return;
  u32 sum[NM],t[NM],tt[NM],y0[NM],y[NM];
  for(int i=0;i<NM;i++) sum[i]=SUMR[(long)r*NM+i]%MODV[i];
  v_scale(sum,sum); v_scale(sum,sum);
  int K=0; for(int j=1;j<=KMAXV;j++){ v_add(t,sum,C_NSJ[j]); K+=v_nonneg(t); }
  v_mul(t,sum,C_T2[K+1]); v_scale(t,t); v_scale(tt,t);
  v_mul(t,tt,C_32); v_scale(t,t); v_sub(y0,C_48,t); d_newton(y,tt,y0);
  memcpy(Y+(long)r*NM,y,sizeof(y)); KR[r]=K;
  u32 tp1[NM],v[NM],U[NM],U2[NM],acc[NM],lnt[NM],lse[NM];
  v_add(tp1,tt,C_S); d_newton(v,tp1,C_175);
  v_add(t,tt,C_NS); v_mul(t,t,v); v_scale(U,t); v_mul(t,U,U); v_scale(U2,t);
  memcpy(acc,C_AT[ATJ],sizeof(acc));
  for(int j=ATJ-1;j>=0;j--){ v_mul(acc,acc,U2); v_scale(acc,acc); v_add(acc,acc,C_AT[j]); }
  v_mul(t,acc,U); v_scale(t,t); v_mulk(lnt,t,2);
  v_mulk(t,C_LN2,(u32)(K+1)); v_add(lse,MX+(long)r*NM,t); v_add(lse,lse,lnt);
  LSE[r]=v_read(lse);
  v_sub(t,lse,X+((long)r*V+tgt[r])*NM); LOSS[r]=v_read(t);
}
__global__ void k_p(int rows,int V,const u32*MN,const u32*Y,const int*KR,i64*P,int*E){
  long g=(long)blockIdx.x*blockDim.x+threadIdx.x; if(g>=(long)rows*V) return; int r=(int)(g/V);
  u32 t[NM]; v_mul(t,MN+g*NM,Y+(long)r*NM); v_scale(t,t); P[g]=v_read(t); E[g]+=KR[r]+1;
}
__global__ void k_fgemm(const signed char*A,const signed char*W,int rows,int K,int V,float sc,float*X){
  int r=blockIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=V) return; float acc=0.f;
  for(int k=0;k<K;k++){ int w=W[(long)j*K+k]; if(!w) continue; acc+= (float)w*(float)A[(long)r*K+k]; }
  X[(long)r*V+j]=acc*sc;
}
__global__ void k_frow(const float*X,int rows,int V,const int*tgt,float*LSE,float*LOSS){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=rows) return; const float*x=X+(long)r*V;
  float mx=x[0]; for(int i=1;i<V;i++) mx=fmaxf(mx,x[i]);
  float s=0.f; for(int i=0;i<V;i++) s+=expf(x[i]-mx);
  float l=mx+logf(s); LSE[r]=l; LOSS[r]=l-x[tgt[r]];
}
__global__ void k_fp(const float*X,int rows,int V,const float*LSE,float*P){
  long g=(long)blockIdx.x*blockDim.x+threadIdx.x; if(g>=(long)rows*V) return; P[g]=expf(X[g]-LSE[g/V]);
}
#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ fprintf(stderr,"CUDA %s at %d\n",cudaGetErrorString(e_),__LINE__); exit(2);} }while(0)
static void* rd(const char*p,size_t n){ FILE*f=fopen(p,"rb"); if(!f){perror(p);exit(3);} void*b=malloc(n); if(fread(b,1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(3);} fclose(f); return b; }
static void wr(const char*p,const void*b,size_t n){ FILE*f=fopen(p,"wb"); fwrite(b,1,n,f); fclose(f); }
int main(int argc,char**argv){
  /* argv: dir rows K V cs rep */
  const char*dir=argv[1]; int rows=atoi(argv[2]),K=atoi(argv[3]),V=atoi(argv[4]); unsigned long long cs=strtoull(argv[5],0,10); int rep=atoi(argv[6]);
  char p[1024]; snprintf(p,1024,"%s/A.bin",dir); signed char*A=(signed char*)rd(p,(size_t)rows*K);
  snprintf(p,1024,"%s/W.bin",dir); signed char*W=(signed char*)rd(p,(size_t)V*K);
  snprintf(p,1024,"%s/tgt.bin",dir); int*tg=(int*)rd(p,rows*4);
  u32 csr[NM]; CSLINE CK(cudaMemcpyToSymbol(C_CS,csr,sizeof(csr)));
  signed char*dA,*dW; int*dT,*dE,*dK,*derr; u32*dX,*dMX,*dMN,*dS,*dY; i64*dP,*dL,*dLo;
  CK(cudaMalloc(&dA,(size_t)rows*K)); CK(cudaMalloc(&dW,(size_t)V*K)); CK(cudaMalloc(&dT,rows*4));
  CK(cudaMalloc(&dX,(size_t)rows*V*NM*4)); CK(cudaMalloc(&dMX,rows*NM*4)); CK(cudaMalloc(&dMN,(size_t)rows*V*NM*4));
  CK(cudaMalloc(&dS,rows*NM*4)); CK(cudaMalloc(&dY,rows*NM*4)); CK(cudaMalloc(&dK,rows*4)); CK(cudaMalloc(&derr,4));
  CK(cudaMalloc(&dE,(size_t)rows*V*4)); CK(cudaMalloc(&dP,(size_t)rows*V*8)); CK(cudaMalloc(&dL,rows*8)); CK(cudaMalloc(&dLo,rows*8));
  CK(cudaMemcpy(dA,A,(size_t)rows*K,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dW,W,(size_t)V*K,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dT,tg,rows*4,cudaMemcpyHostToDevice));
  i64*P=(i64*)malloc((size_t)rows*V*8),*L=(i64*)malloc(rows*8),*Lo=(i64*)malloc(rows*8); int*E=(int*)malloc((size_t)rows*V*4);
  cudaEvent_t e0,e1,ek[4]; cudaEventCreate(&e0); cudaEventCreate(&e1); for(int q=0;q<4;q++) cudaEventCreate(&ek[q]);
  float ms_best=1e30f, kbest[5]={1e30f,1e30f,1e30f,1e30f,1e30f};
  long n=(long)rows*V; int tb=128;
  for(int it=0;it<rep;it++){
    CK(cudaMemset(dS,0,rows*NM*4)); CK(cudaMemset(derr,0,4));
    cudaEventRecord(e0);
    k_gemm<<<dim3((V+127)/128,rows),128>>>(dA,dW,rows,K,V,dX); cudaEventRecord(ek[0]);
    k_max<<<rows,MAXTB>>>(dX,rows,V,dMX); cudaEventRecord(ek[1]);
    k_exp<<<(int)((n+tb-1)/tb),tb>>>(dX,rows,V,dMX,dMN,dE,dS,derr); cudaEventRecord(ek[2]);
    k_row<<<(rows+63)/64,64>>>(dX,rows,V,dT,dMX,dS,dY,dK,dL,dLo); cudaEventRecord(ek[3]);
    k_p<<<(int)((n+tb-1)/tb),tb>>>(rows,V,dMN,dY,dK,dP,dE);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms,e0,e1); if(ms<ms_best) ms_best=ms;
    { cudaEvent_t pv=e0; for(int q=0;q<4;q++){ float t; cudaEventElapsedTime(&t,pv,ek[q]); if(t<kbest[q]) kbest[q]=t; pv=ek[q]; }
      float t; cudaEventElapsedTime(&t,ek[3],e1); if(t<kbest[4]) kbest[4]=t; }
    int herr; CK(cudaMemcpy(&herr,derr,4,cudaMemcpyDeviceToHost)); if(herr){ fprintf(stderr,"ring range error\n"); return 4; }
    CK(cudaMemcpy(P,dP,(size_t)n*8,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(E,dE,(size_t)n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(L,dL,rows*8,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(Lo,dLo,rows*8,cudaMemcpyDeviceToHost));
    snprintf(p,1024,"%s/out_P_%d.bin",dir,it); wr(p,P,(size_t)n*8); snprintf(p,1024,"%s/out_E_%d.bin",dir,it); wr(p,E,(size_t)n*4);
    snprintf(p,1024,"%s/out_L_%d.bin",dir,it); wr(p,L,rows*8); snprintf(p,1024,"%s/out_Lo_%d.bin",dir,it); wr(p,Lo,rows*8);
  }
  /* ABACUS-NL-1 step 1: fp32 softmax/LSE/loss on the same inputs, same GPU (expf/logf) */
  float*fX,*fL,*fLo,*fP; float sc=(float)((double)cs/(double)SV); float f_best=1e30f;
  CK(cudaMalloc(&fX,(size_t)n*4)); CK(cudaMalloc(&fP,(size_t)n*4)); CK(cudaMalloc(&fL,rows*4)); CK(cudaMalloc(&fLo,rows*4));
  float*hL=(float*)malloc(rows*4),*hLo=(float*)malloc(rows*4);
  for(int it=0;it<rep;it++){
    cudaEventRecord(e0);
    k_fgemm<<<dim3((V+127)/128,rows),128>>>(dA,dW,rows,K,V,sc,fX);
    k_frow<<<(rows+63)/64,64>>>(fX,rows,V,dT,fL,fLo);
    k_fp<<<(int)((n+tb-1)/tb),tb>>>(fX,rows,V,fL,fP);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms,e0,e1); if(ms<f_best) f_best=ms;
  }
  CK(cudaMemcpy(hL,fL,rows*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hLo,fLo,rows*4,cudaMemcpyDeviceToHost));
  snprintf(p,1024,"%s/out_fL.bin",dir); wr(p,hL,rows*4); snprintf(p,1024,"%s/out_fLo.bin",dir); wr(p,hLo,rows*4);
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
  printf("{\"gpu\":\"%s\",\"rows\":%d,\"K\":%d,\"V\":%d,\"best_ms\":%.3f,\"f32_best_ms\":%.3f,\"reps\":%d,\"kernels_ms\":{\"gemm\":%.3f,\"max\":%.3f,\"exp\":%.3f,\"row\":%.3f,\"p\":%.3f}}\n",pr.name,rows,K,V,ms_best,f_best,rep,kbest[0],kbest[1],kbest[2],kbest[3],kbest[4]);
  return 0;
}
""".replace("FINLINE", fin).replace("EXPD", str(EXP_D)).replace("ZMAXV", str(ZMAX)).replace("SV)", f"{S}.0)") \
   .replace("NEWTONV", str(NEWTON)).replace("KMAXV", str(KMAX)).replace("ATJ", str(ATANH_J)) \
   .replace("MODV[i]", "(u32)(" + "".join(f"i=={i}?{m}u:" for i, m in enumerate(MODS)) + "0u)") \
   .replace("CSLINE", " ".join(f"csr[{i}]=(u32)(cs%{m}ull);" for i, m in enumerate(MODS)))

def prep(d):
    os.makedirs(d, exist_ok=True); k = J.Kernel(); meta = {}
    for name, rows, K, V, seed in SHAPES:
        sd = os.path.join(d, name); os.makedirs(sd, exist_ok=True)
        A, W, tgt = make_inputs(rows, K, V, seed)
        A.tofile(os.path.join(sd, "A.bin")); W.tofile(os.path.join(sd, "W.bin")); tgt.tofile(os.path.join(sd, "tgt.bin"))
        P, E, LSE, LOSS, dt = cpu_twin(A, W, tgt, K, k)
        np.save(os.path.join(sd, "cpu_P.npy"), P); np.save(os.path.join(sd, "cpu_E.npy"), E)
        np.save(os.path.join(sd, "cpu_L.npy"), LSE); np.save(os.path.join(sd, "cpu_Lo.npy"), LOSS)
        meta[name] = {"rows": rows, "K": K, "V": V, "cs": cs_of(K), "cpu_softmax_s": round(dt, 3),
                      "mean_loss": float(np.mean(LOSS)) / S}
    open(os.path.join(d, "ring_nl.cu"), "w").write(render_cu())
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1); print(json.dumps(meta, indent=1))

def check(d, reps=3):
    meta = json.load(open(os.path.join(d, "meta.json"))); out = {}
    for name, m in meta.items():
        sd = os.path.join(d, name); rows, V = m["rows"], m["V"]
        ref = {k: np.load(os.path.join(sd, f"cpu_{k}.npy")) for k in ("P", "E", "L", "Lo")}
        mism = []; runs = []
        for it in range(reps):
            g = {"P": np.fromfile(os.path.join(sd, f"out_P_{it}.bin"), np.int64).reshape(rows, V),
                 "E": np.fromfile(os.path.join(sd, f"out_E_{it}.bin"), np.int32).reshape(rows, V),
                 "L": np.fromfile(os.path.join(sd, f"out_L_{it}.bin"), np.int64),
                 "Lo": np.fromfile(os.path.join(sd, f"out_Lo_{it}.bin"), np.int64)}
            mism.append(int(sum((g[k] != ref[k]).sum() for k in g))); runs.append(g)
        same = all(all((runs[i][k] == runs[0][k]).all() for k in runs[0]) for i in range(1, reps))
        out[name] = {**m, "G1_mismatches_per_run": mism, "G2_runs_identical": same}
        fpath = os.path.join(sd, "out_fLo.bin")
        if os.path.exists(fpath):                        # ABACUS-NL-1 G4: ring vs fp32, both against fp64 on the same logits
            A, W, tgt = make_inputs(rows, m["K"], V, dict((s[0], s[4]) for s in SHAPES)[name])
            x = (A.astype(np.int64) @ W.astype(np.int64).T).astype(np.float64) * m["cs"] / S
            mx = x.max(1); lse64 = mx + np.log(np.exp(x - mx[:, None]).sum(1)); loss64 = lse64 - x[np.arange(rows), tgt]
            fL = np.fromfile(os.path.join(sd, "out_fL.bin"), np.float32).astype(np.float64)
            fLo = np.fromfile(fpath, np.float32).astype(np.float64)
            err = lambda a, b: float(np.max(np.abs(a - b)))
            out[name]["G4_vs_fp64"] = {"ring_lse_maxabs": err(ref["L"] / S, lse64), "ring_loss_maxabs": err(ref["Lo"] / S, loss64),
                                       "f32_lse_maxabs": err(fL, lse64), "f32_loss_maxabs": err(fLo, loss64)}
    gpu = json.load(open(os.path.join(d, "gpu.json"))) if os.path.exists(os.path.join(d, "gpu.json")) else {}
    for name in out: out[name]["G3_gpu"] = gpu.get(name)
    R = {"date": "2026-10-13", "shapes": out,
         "verdict": "PASS" if all(all(x == 0 for x in v["G1_mismatches_per_run"]) and v["G2_runs_identical"] for v in out.values()) else "FAIL"}
    json.dump(R, open(os.path.join(HERE, "gpu_nl.result.json"), "w"), indent=1); print(json.dumps(R, indent=1))

if __name__ == "__main__":
    {"prep": prep, "check": check, "cu": lambda d: open(os.path.join(d, "ring_nl.cu"), "w").write(render_cu())}[sys.argv[1]](sys.argv[2])
