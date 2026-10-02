"""Rebuild the mlx-omarchy mlx-lm patch series on mlx-lm 94cdcae (the 0.32 API line oMLX pins).

Each step edits a staged copy of upstream in series order; each step's diff becomes one patch file.
Edits are exact-match replacements that refuse on any mismatch.
"""
import shutil
import subprocess
import sys
from pathlib import Path

UP = Path(sys.argv[1])          # pristine .../mlx-lm-<sha>
OLD = Path(sys.argv[2])         # repo patches/ (0.31.3 series), for the steps that still apply clean
OUT = Path(sys.argv[3])         # repo patches/mlx-lm-0.32/
ROOT = Path("/tmp/mlxlm032/rebuild")
shutil.rmtree(ROOT, ignore_errors=True)
(ROOT / "b").mkdir(parents=True)
shutil.copytree(UP / "mlx_lm", ROOT / "b/mlx_lm")
OUT.mkdir(parents=True, exist_ok=True)


def edit(rel, old, new):
    f = ROOT / "b/mlx_lm" / rel
    text = f.read_text()
    if text.count(old) != 1:
        sys.exit(f"{rel}: expected exactly one match, found {text.count(old)}:\n{old}")
    f.write_text(text.replace(old, new, 1))


def clean(name):
    subprocess.run(["patch", "-s", "-d", str(ROOT / "b"), "-p1", "--forward", "--fuzz=0", "--no-backup-if-mismatch",
                    "-i", str(OLD / f"{name}.patch")], check=True)


def step(name, fn):
    shutil.rmtree(ROOT / "a", ignore_errors=True)
    shutil.copytree(ROOT / "b", ROOT / "a")
    fn()
    d = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.orig", "-x", "*.rej", "a/mlx_lm", "b/mlx_lm"], cwd=ROOT,
                       capture_output=True, text=True)
    if not d.stdout:
        sys.exit(f"{name}: produced no change")
    lines = [ln.split("\t")[0] if ln.startswith(("--- ", "+++ ")) else ln for ln in d.stdout.splitlines()]
    (OUT / f"{name}.patch").write_text("\n".join(ln for ln in lines if not ln.startswith("diff ")) + "\n")
    print("wrote", name)


DISPATCH = """    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
"""
FAST = """    if not use_kernel or mx.default_device() != mx.gpu:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    if hasattr(mx.fast, "gated_delta_update"):
        out, st = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
        return out, st
    if not mx.metal.is_available() or k.shape[-1] < 32 or k.shape[-1] % 32 != 0:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
"""
step("mlx-lm-gated-delta-fast-route", lambda: edit("models/gated_delta.py", DISPATCH, FAST))

REPEAT = """    # Prefill (T > 1) with Hk != Hv: the fused backend needs Hk == Hv
    # (omarchy/primitives.cpp use_fallback). Without the repeat it falls
    # into a per-token loop. Decode (T == 1) uses the raw route, which
    # repeats inside the backend, so it must not repeat here.
    Hk, Hv = q.shape[-2], v.shape[-2]
    if Hv != Hk and q.shape[1] > 1:
        q = mx.repeat(q, Hv // Hk, -2)
        k = mx.repeat(k, Hv // Hk, -2)

"""
step("mlx-lm-gated-delta-fast-route-repeat",
     lambda: edit("models/gated_delta.py", "    if not use_kernel or mx.default_device() != mx.gpu:\n",
                  REPEAT + "    if not use_kernel or mx.default_device() != mx.gpu:\n"))

RAW_OLD = """    if hasattr(mx.fast, "gated_delta_update"):
        out, st = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
"""
# The raw kernel derives g and beta itself with the default formula, so the
# lower_bound and allow_neg_eigval variants (new in this mlx-lm line) keep the
# composed g/beta path.
RAW_NEW = """    if hasattr(mx.fast, "gated_delta_update"):
        if (
            q.shape[1] == 1
            and lower_bound is None
            and not allow_neg_eigval
            and hasattr(mx.fast, "gated_delta_update_raw")
        ):
            return mx.fast.gated_delta_update_raw(
                q, k, v, a, b, A_log, dt_bias, state, mask
            )
        out, st = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
"""
step("mlx-lm-gated-delta-raw", lambda: edit("models/gated_delta.py", RAW_OLD, RAW_NEW))


def greedy():
    old = (OLD / "mlx-lm-greedy-prune.patch").read_text()
    body = old.split("+def _greedy_head(model):", 1)[1].split("\n def generate_step(", 1)[0]
    head = "def _greedy_head(model):" + "\n".join(ln[1:] for ln in body.splitlines()).rstrip() + "\n\n\n"
    edit("generate.py", "def generate_step(\n", head + "def generate_step(\n")
    edit("generate.py", "    sampler = sampler or greedy_sampler\n\n    def _model_call(",
         "    greedy = _greedy_head(model) if sampler is None and not logits_processors else None\n"
         "    sampler = sampler or greedy_sampler\n\n    def _model_call(")
    edit("generate.py", """        with mx.stream(stream):
            logits = _model_call(
                input_tokens=input_tokens[None],""",
         """        with mx.stream(stream):
            if greedy and input_embeddings is None and input_tokens.size == 1:
                hidden = greedy.body(input_tokens[None], cache=prompt_cache)
                quantize_cache_fn(prompt_cache)
                return greedy(hidden[:, -1, :])

            logits = _model_call(
                input_tokens=input_tokens[None],""")
    edit("generate.py", """        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)
        mx.async_eval(y, logprobs)
""", """        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)
        if greedy:
            mx.async_eval(y)
        else:
            mx.async_eval(y, logprobs)
""")
    edit("generate.py", """                next_y, next_logprobs = _step(y)
                mx.async_eval(next_y, next_logprobs)
            if n == 0:
                mx.eval(y)
                prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
            if n == max_tokens:
                break
            yield y.item(), logprobs
            if n % 256 == 0:""", """                next_y, next_logprobs = _step(y)
                if greedy:
                    mx.async_eval(next_y)
                else:
                    mx.async_eval(next_y, next_logprobs)
            if n == 0:
                mx.eval(y)
                prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
            if n == max_tokens:
                break
            yield y.item(), logprobs
            if n % 256 == 0:""")


step("mlx-lm-greedy-prune", greedy)

QK_OLD = """    rms_eps = eps * inv_scale**2
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, rms_eps)
    k = inv_scale * mx.fast.rms_norm(k, None, rms_eps)
    return q, k
"""
QK_NEW = """    rms_eps = eps * inv_scale**2
    if q.dtype == mx.bfloat16 and hasattr(mx.fast, "rms_norm_scaled"):
        q = mx.fast.rms_norm_scaled(q, None, inv_scale * inv_scale, rms_eps)
        k = mx.fast.rms_norm_scaled(k, None, inv_scale, rms_eps)
        return q, k
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, rms_eps)
    k = inv_scale * mx.fast.rms_norm(k, None, rms_eps)
    return q, k
"""
step("mlx-lm-qwen35-qk-scaled", lambda: edit("models/gated_delta.py", QK_OLD, QK_NEW))
step("mlx-lm-qwen35-gdn-conv", lambda: clean("mlx-lm-qwen35-gdn-conv"))
step("mlx-lm-conv-silu", lambda: clean("mlx-lm-conv-silu"))

GN_OLD = """        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return precise_swiglu(hidden_states, gate, x)
"""
GN_NEW = """        if (
            gate is not None
            and hidden_states.dtype == mx.bfloat16
            and mx.default_device() == mx.gpu
            and hasattr(mx.fast, "rms_norm_gated")
        ):
            return mx.fast.rms_norm_gated(
                hidden_states, gate, self.weight, self.eps
            )
""" + GN_OLD
step("mlx-lm-qwen35-gated-norm", lambda: edit("models/qwen3_next.py", GN_OLD, GN_NEW))


def ttft():
    edit("generate.py", "import json\nimport sys\n", "import json\nimport os\nimport sys\n")
    edit("generate.py", """    with mx.stream(stream):
        total_prompt_tokens = (""", """    # Early first submit (mlx-omarchy): prompt graphs and the first decode
    # step commit their first batch after 128 nodes so the GPU starts while the
    # host is still recording (host record is fully exposed until the first
    # token). Cleared right after the first token so pipelined decode keeps one
    # submit per token. Scheduling only; backends that ignore the variable are
    # unaffected. MLX_OMARCHY_NO_TTFT_EARLY_SUBMIT=1 disables.
    _omarchy_early = (
        "MLX_OMARCHY_BATCH_FIRST" not in os.environ
        and "MLX_OMARCHY_NO_TTFT_EARLY_SUBMIT" not in os.environ
    )
    if _omarchy_early:
        os.environ["MLX_OMARCHY_BATCH_FIRST"] = "128"

    with mx.stream(stream):
        total_prompt_tokens = (""")
    edit("generate.py", """            if n == 0:
                mx.eval(y)
                prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
            if n == max_tokens:
                break
            yield y.item(), logprobs""", """            if n == 0:
                mx.eval(y)
                if _omarchy_early:
                    os.environ.pop("MLX_OMARCHY_BATCH_FIRST", None)
                prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
            if n == max_tokens:
                break
            yield y.item(), logprobs""")


step("mlx-lm-ttft-early-submit", ttft)
step("mlx-lm-last-logits", lambda: clean("mlx-lm-last-logits"))
shutil.copytree(ROOT / "b/mlx_lm", ROOT / "patched_mlx_lm", dirs_exist_ok=True)
print("staged patched tree:", ROOT / "patched_mlx_lm")
