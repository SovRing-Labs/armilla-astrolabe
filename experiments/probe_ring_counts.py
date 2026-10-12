import math, time, random, json
import numpy as np
def odd_primes(n):
    out=[];k=3
    while len(out)<n:
        if all(k%p for p in range(3,int(k**.5)+1,2)): out.append(k)
        k+=2
    return out
res={}
# 1. range by ring count
for n in (5,15,25,50,53,55,64,100,128):
    P=odd_primes(n); res[f"T{n}"]={"max_modulus":P[-1],"bits":round(sum(math.log2(p) for p in P),1),"fits_uint8":P[-1]<256}
# 8-bit-lane maximum: largest power of each prime <=256
pp=[]
for p in [2]+odd_primes(53):
    q=p
    while q*p<=256: q*=p
    pp.append(q)
res["uint8_prime_power_set"]={"count":len(pp),"bits":round(sum(math.log2(q) for q in pp),1),"moduli_head":pp[:6]}
# 2. faithful evolved Gelosia: limbs in the lattice, each box an RNS word, one carry pass at the end
P=odd_primes(50); m=np.array(P,dtype=object); M=math.prod(P)
def to_rns(x): return np.array([x%p for p in P],dtype=np.int64)
k=[pow(M//p,-1,p) for p in P]
def from_rns(r): return sum(int(ri)*(M//p)*ki for ri,p,ki in zip(r,P,k))%M
def gelosia_mul(a,b,limb_bits):
    B=1<<limb_bits
    A=[];x=a
    while x: A.append(x&(B-1)); x>>=limb_bits
    Bv=[];x=b
    while x: Bv.append(x&(B-1)); x>>=limb_bits
    K=max(len(A),len(Bv))
    assert K*(B-1)**2 < M, "diagonal would wrap"
    mods=np.array(P,dtype=np.int64)
    RA=np.stack([to_rns(v) for v in A]); RB=np.stack([to_rns(v) for v in Bv])
    boxes=(RA[:,None,:]*RB[None,:,:])%mods          # every lattice box, carry-free, independent
    diags=[]
    for d in range(len(A)+len(Bv)-1):                 # sum each diagonal inside the rings
        i0=max(0,d-len(Bv)+1); i1=min(d,len(A)-1)
        s=np.zeros(len(P),dtype=np.int64)
        for i in range(i0,i1+1): s=(s+boxes[i,d-i])%mods
        diags.append(from_rns(s))                     # one read-out per diagonal
    out=0                                             # the single carry pass, as on paper
    for d,v in enumerate(diags): out+=v<<(d*limb_bits)
    return out
random.seed(1)
a=random.getrandbits(8000)|1<<7999; b=random.getrandbits(8000)|1<<7999
for lb in (128,150):
    t=time.perf_counter(); g=gelosia_mul(a,b,lb); tg=time.perf_counter()-t
    t=time.perf_counter(); [a*b for _ in range(200)]; tb=(time.perf_counter()-t)/200
    res[f"gelosia_8000bit_limb{lb}"]={"exact":g==a*b,"limbs":math.ceil(8000/lb),"gelosia_s":round(tg,4),"python_bigint_s":tb,"ratio_bigint_faster":round(tg/tb,1)}
print(json.dumps(res,indent=1))
