import mlx.core as mx, numpy as np
mx.random.seed(0)
K,N=4096,4096; g=64; bits=4
w=mx.random.randint(0,15,(N,K//8)).astype(mx.uint32)
s=mx.random.normal((N,K//g)).astype(mx.bfloat16)*0.01
b=mx.random.normal((N,K//g)).astype(mx.bfloat16)*0.01
mx.eval(w,s,b)
for M in (2,4,8):
    x=mx.random.normal((M,K)).astype(mx.bfloat16); mx.eval(x)
    yb=mx.quantized_matmul(x,w,s,b,transpose=True,group_size=g,bits=bits); mx.eval(yb)
    ok=True; md=0.0
    for i in range(M):
        yi=mx.quantized_matmul(x[i:i+1],w,s,b,transpose=True,group_size=g,bits=bits); mx.eval(yi)
        a=np.array(yb[i].astype(mx.float32)); c=np.array(yi[0].astype(mx.float32))
        if not (a==c).all(): ok=False
        md=max(md, float(np.abs(a-c).max()))
    print(f"M={M} width_invariant_vs_M1 {ok} maxdiff {md}")
