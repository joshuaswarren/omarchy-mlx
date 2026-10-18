"""gdn_logits.py: per-step log-probabilities of a pinned greedy decode, for the GDN decode-kernel numerics gate.
One process is one arm (the caller sets the environment). Records, per generated token: the id, the top 20 (id, logprob),
and the full float32 log-probability vector for the first --full-steps tokens. Same prompt handling as scripts/bench_decode.py."""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


def record(resp, ids, top_i, top_v, full, full_steps, to_np):
    a = to_np(resp.logprobs)
    ids.append(int(resp.token))
    idx = np.argpartition(-a, 20)[:20]
    idx = idx[np.argsort(-a[idx], kind="stable")]
    top_i.append(idx)
    top_v.append(a[idx])
    if len(ids) <= full_steps:
        full.append(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--full-steps", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wheel", required=True)
    ap.add_argument("--scripts", required=True, help="the scripts dir holding mlx_provenance.py and caps_sim_guard.py")
    args = ap.parse_args()
    sys.path.insert(0, args.scripts)
    import caps_sim_guard
    caps_sim_guard.refuse_if_simulated("gdn_logits")
    from mlx_provenance import ProvenanceRefusal, installed_provenance, provenance_line
    try:
        prov = installed_provenance(expect_wheel=Path(args.wheel))
        if prov["verified"] != "match":
            raise ProvenanceRefusal(f"provenance {prov['verified']}: {prov.get('mismatch')}")
    except ProvenanceRefusal as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        sys.exit(3)
    print(provenance_line(prov), flush=True)
    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    model, tokenizer = load(args.model)
    mx.random.seed(0)
    sampler = make_sampler(temp=0.0)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True)
    if isinstance(prompt, str):
        bos = getattr(tokenizer, "bos_token", None)
        prompt = tokenizer.encode(prompt, add_special_tokens=(bos is None or not prompt.startswith(bos)))
    prompt_ids = mx.array(prompt)
    tokenizer.eos_token_ids = set()
    det = tokenizer.detokenizer
    tokenizer._detokenizer_class = lambda _: det
    det.reset()
    ids, top_i, top_v, full = [], [], [], []

    def to_np(lp):
        lp = lp.astype(mx.float32)
        mx.eval(lp)
        return np.array(lp)

    for resp in stream_generate(model, tokenizer, prompt_ids, max_tokens=args.tokens, sampler=sampler):
        record(resp, ids, top_i, top_v, full, args.full_steps, to_np)
    if len(ids) != args.tokens:
        print(f"FAIL: {len(ids)} tokens, {args.tokens} requested", file=sys.stderr)
        sys.exit(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "top.npz", ids=np.array(ids), top_i=np.stack(top_i), top_v=np.stack(top_v))
    np.save(out / "full.npy", np.stack(full))
    dig = hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()[:16]
    print(f"logits-run ids_sha256_16 {dig} n={len(ids)} prompt_tokens={int(prompt_ids.size)} vocab={full[0].shape[0]} full_steps={len(full)}")


if __name__ == "__main__":
    main()
