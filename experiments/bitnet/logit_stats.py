# logits of the Armilla run vs bitnet.cpp, compared only while both streams share the same context (up to and including the first differing token)
import array
V=128256
def ld(f):
    x=array.array('f');x.frombytes(open(f,'rb').read());return x
def toks(f): return open(f).read().split()
for k in 'ABC':
    rt=toks(f'ref{k}.tok')
    for name in ('arm','card'):
        at=toks(f'{name}{k}.tok'); n=next((i for i,(x,y) in enumerate(zip(rt,at)) if x!=y),len(rt)); upto=min(n+1,len(rt))
        R=ld(f'ref{k}.bin');A=ld(f'{name}{k}.bin');md=0;ma=0.0
        for s in range(upto):
            r=R[s*V:(s+1)*V];a=A[s*V:(s+1)*V];d=max(abs(x-y) for x,y in zip(r,a));md=max(md,d);ma+=sum(abs(x-y) for x,y in zip(r,a))/V
        print(f'prompt {k} {name}: first token difference at step {"none (64/64 identical)" if n==len(rt) else n}; over {upto} shared-context steps max|logit diff| {md:.3f}, mean|diff| {ma/upto:.4f}')
