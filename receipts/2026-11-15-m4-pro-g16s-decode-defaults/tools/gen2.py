import os,sys,time,hashlib
os.environ["MLX_OMARCHY_NO_MLXLM_WARNING"]="1"
import mlx.core as mx
from mlx_lm import load, stream_generate
repo=sys.argv[1]; ntok=int(sys.argv[2]) if len(sys.argv)>2 else 128
m,tok=load(repo)
long="Summarize the following text. "+("The quick brown fox jumps over the lazy dog near the river bank. "*20)
long2k="Summarize the following text. "+("The quick brown fox jumps over the lazy dog near the river bank while the farmer counts his sheep. "*100)
p=tok.apply_chat_template([{"role":"user","content":"Explain in three sentences why the sky is blue."}],add_generation_prompt=True)
for r in stream_generate(m,tok,tok.apply_chat_template([{"role":"user","content":long}],add_generation_prompt=True),max_tokens=4): pass
res={}
for name,pr in (("pf300",long),("pf2k",long2k)):
    ids=tok.apply_chat_template([{"role":"user","content":pr}],add_generation_prompt=True)
    for r in stream_generate(m,tok,ids,max_tokens=1): pw=r
    res[name]=(pw.prompt_tokens,pw.prompt_tps)
txt=""
for r in stream_generate(m,tok,p,max_tokens=ntok):
    last=r; txt+=r.text
print("RESULT",repo.split("/")[-1],"pf300 %d %.1f"%res["pf300"],"pf2k %d %.1f"%res["pf2k"],"decode %d %.2f"%(last.generation_tokens,last.generation_tps),"peak %.2f"%last.peak_memory,"text_sha",hashlib.sha256(txt.encode()).hexdigest()[:12])
print("TEXT:",txt.replace("\n"," ")[:160])
