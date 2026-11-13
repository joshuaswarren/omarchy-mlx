import sys,numpy as np
sys.path.insert(0, 'tools/kernel_recheck')  # run from the repo root
from defs import bf16
f32=np.float32
d=np.load('kda_glue_pre-io.npz'); c=12288
g=lambda k:d['in_'+k].astype(f32); hp,s0,s1,s2,wq,wk,wv=[g(k) for k in "hp s0 s1 s2 wq wk wv".split()]
cs=d['in_cons'].astype(f32)
b=lambda x: bf16(np.asarray(x,dtype=f32)).astype(f32)
def conv(s,w,o,fma=False):
    taps=[s[:c],s[c:2*c],s[2*c:3*c],hp[o:o+c]]; W=w.reshape(c,4); a=np.zeros(c,f32)
    for t in range(4):
        if fma: a=(taps[t].astype(np.float64)*W[:,t]+a).astype(f32)
        else: a=a+taps[t]*W[:,t]
    return b(a)
def sig(x):
    e=b(np.exp(np.abs(x))); den=b(f32(1)+e); y=b(f32(1)/den); return np.where(x<0,y,b(f32(1)-y))
def q_out(fma,rs):
    cq=conv(s0,wq,0,fma); yq=b(cq*sig(cq)).reshape(96,128)
    p=(yq*yq).reshape(96,4,32).sum(2); m=(p[:,0]+p[:,1]+p[:,2]+p[:,3])/f32(128)+cs[3]
    r=(f32(1)/np.sqrt(m)) if rs==0 else (f32(1)/np.sqrt(m))*(f32(1)+f32(2**-22))
    return b(b(yq*r[:,None])*cs[2]).reshape(-1)
dev=d['out_q'].reshape(-1)
for fma in (0,1):
  for rs in (0,1):
    print("fma",fma,"rsqrt+1e-7rel",rs,"mismatch vs device:",(q_out(fma,rs)!=dev).sum())
ref=d['ref_0'].reshape(-1).astype(f32); e=q_out(0,0)
print("dev!=ref",(dev!=ref).sum(),"emu!=ref",(e!=ref).sum(),"dev!=emu",(dev!=e).sum())
print(dev[:6],ref[:6],e[:6], d['in_cons'], d['in_hp'].dtype, d['out_q'].dtype)
print("----")
import itertools
names=["conv","sigin","yq","nq","out"]
res=[]
for mask in itertools.product((0,1),repeat=5):
    rd=dict(zip(names,mask))
    B=lambda x,n: b(x) if rd[n] else np.asarray(x,dtype=f32)
    a=np.zeros(c,f32); W=wq.reshape(c,4); taps=[s0[:c],s0[c:2*c],s0[2*c:3*c],hp[0:c]]
    for t in range(4): a=a+taps[t]*W[:,t]
    cq=B(a,"conv")
    x=cq; e_=np.exp(np.abs(x)); e_=B(e_,"sigin"); den=B(f32(1)+e_,"sigin"); y=B(f32(1)/den,"sigin"); sg=np.where(x<0,y,B(f32(1)-y,"sigin"))
    yq=B(cq*sg,"yq").reshape(96,128)
    p=(yq*yq).reshape(96,4,32).sum(2); m=(p[:,0]+p[:,1]+p[:,2]+p[:,3])/f32(128)+cs[3]
    n=B(yq*(f32(1)/np.sqrt(m))[:,None],"nq"); o=B(n*cs[2],"out").reshape(-1)
    res.append(((o!=dev).sum(),mask))
res.sort(); print(res[:4])
