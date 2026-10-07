#!/usr/bin/env python3
"""Greedy batches decode through the pruned greedy head (batched serve-path head fix).

Root cause (receipts/2026-10-07-batch-decode-2): GenerationBatch._step (the
mlx-lm BatchGenerator that oMLX serves through) always projects every row
onto the full tied vocabulary and evaluates the logprobs every step. The
pruned greedy head (mx.fast.greedy_quantized_argmax, mlx-lm-greedy-prune
patch) served only generate_step, one row. At B=4 on Qwen3.8-2B the head
cost 1.38 -> 4.28 ms per step.

When every row samples greedily (sampler is greedy_sampler) and no row has
logits processors, the step takes each row's token from the greedy head (one
dispatch per row; the token equals argmax of the full logits, proof in
receipts/2026-09-23-vocab-prune.md) and leaves the logprobs lazy: the full
projection runs only if a consumer evaluates them (oMLX drops them unless the
request asked for logprobs). Any other batch keeps the upstream step. A batch
merged from lazy and eager parts evaluates its logprobs (the safe default).

Needs the mlx-lm-greedy-prune patch (_greedy_head). 0.32 line only (no
GenerationBatch in 0.31.3: nothing to patch). Kill switch at runtime:
MLX_OMARCHY_BATCH_GREEDY=0 (read once at import).
Usage: python3 patch-mlx-lm-batch-greedy-head.py /path/to/venv
"""
import ast
import glob
import sys

MARKER = "MLX_OMARCHY_BATCH_GREEDY"

HEAD_OLD = """    def head(hidden):
        token, _ = mx.fast.greedy_quantized_argmax(
            hidden, emb.weight, emb.scales, emb.biases, high, group_size=64, bits=4
        )
        logits = emb.as_linear(hidden)
        return token, (logits - mx.logsumexp(logits, keepdims=True)).squeeze(0)

    head.body = text.model
    return head
"""
HEAD_NEW = """    def token(hidden):
        return mx.fast.greedy_quantized_argmax(
            hidden, emb.weight, emb.scales, emb.biases, high, group_size=64, bits=4
        )[0]

    def head(hidden):
        logits = emb.as_linear(hidden)
        return token(hidden), (logits - mx.logsumexp(logits, keepdims=True)).squeeze(0)

    head.body = text.model
    head.token = token
    head.logits = emb.as_linear
    return head


# mlx-omarchy: greedy batches decode through the pruned greedy head with lazy
# logprobs (GenerationBatch._step). MLX_OMARCHY_BATCH_GREEDY=0 keeps the
# full-logits step.
_BATCH_GREEDY = os.environ.get("MLX_OMARCHY_BATCH_GREEDY", "1") != "0"
_UNSET = object()
"""

INIT_OLD = """        self._matchers = [ss.matcher() for ss in stop_sequences]

        if self.uids:
            self._step()
"""
INIT_NEW = """        self._matchers = [ss.matcher() for ss in stop_sequences]
        self._greedy = _UNSET
        # An empty logprob list is trivially lazy, so an empty batch never
        # forces an eval on the batches extended into it.
        self._current_lazy = True
        self._next_lazy = True

        if self.uids:
            self._step()
"""

EXTEND_OLD = """        self._num_tokens.extend(batch._num_tokens)
        self._matchers.extend(batch._matchers)

    def _step(self) -> Tuple[List[int], List[mx.array]]:
"""
EXTEND_NEW = """        self._num_tokens.extend(batch._num_tokens)
        self._matchers.extend(batch._matchers)
        self._current_lazy = self._current_lazy and batch._current_lazy
        self._next_lazy = self._next_lazy and batch._next_lazy

    def _batch_greedy_head(self):
        \"\"\"The greedy head when every row samples greedily without logits
        processors, else None.\"\"\"
        if not _BATCH_GREEDY or any(self.logits_processors):
            return None
        samplers = self.samplers or [None] * len(self.uids)
        if any((self.fallback_sampler if s is None else s) is not greedy_sampler
               for s in samplers):
            return None
        if self._greedy is _UNSET:
            self._greedy = _greedy_head(self.model)
        return self._greedy

    def _step(self) -> Tuple[List[int], List[mx.array]]:
"""

STEP_OLD = """        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens

        # Forward pass
        logits = self.model(inputs[:, None], cache=self.prompt_cache)
"""
STEP_NEW = """        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        self._current_lazy = self._next_lazy
        inputs = self._current_tokens

        greedy = self._batch_greedy_head()
        if greedy is not None:
            hidden = greedy.body(inputs[:, None], cache=self.prompt_cache)[:, -1, :]
            self._next_tokens = mx.concatenate(
                [greedy.token(hidden[e : e + 1]) for e in range(hidden.shape[0])]
            )
            logits = greedy.logits(hidden)
            self._next_logprobs = list(
                logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            )
            self._next_lazy = True
            mx.async_eval(self._next_tokens)
            mx.eval(inputs, [] if self._current_lazy else self._current_logprobs)
            inputs = inputs.tolist()
            for sti, ti in zip(self.tokens, inputs):
                sti.append(ti)
            return inputs, self._current_logprobs
        self._next_lazy = False

        # Forward pass
        logits = self.model(inputs[:, None], cache=self.prompt_cache)
"""

EVAL_OLD = """        mx.eval(inputs, self._current_logprobs)
        inputs = inputs.tolist()
"""
EVAL_NEW = """        mx.eval(inputs, [] if self._current_lazy else self._current_logprobs)
        inputs = inputs.tolist()
"""

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm")
if not site:
    sys.exit("mlx_lm not found under " + venv)
q = site[0] + "/generate.py"
text = open(q).read()
if MARKER in text:
    print("already patched:", q)
    sys.exit(0)
if "class GenerationBatch:" not in text:
    print("no GenerationBatch (mlx-lm 0.31 line); nothing to patch:", q)
    sys.exit(0)
if "def _greedy_head(model):" not in text:
    sys.exit("generate.py lacks _greedy_head (apply mlx-lm-greedy-prune.patch first)")
if "\nimport os\n" not in text:
    text = text.replace("\nimport ", "\nimport os\nimport ", 1)
start = text.find("class GenerationBatch:")
end = text.find("\nclass ", start + 1)
head, body, tail = text[:start], text[start:end], text[end:]
if head.count(HEAD_OLD) != 1:
    sys.exit("unrecognized _greedy_head in " + q + "; refusing to patch")
head = head.replace(HEAD_OLD, HEAD_NEW, 1)
for old, new in [(INIT_OLD, INIT_NEW), (EXTEND_OLD, EXTEND_NEW), (STEP_OLD, STEP_NEW),
                 (EVAL_OLD, EVAL_NEW)]:
    if body.count(old) != 1:
        sys.exit("unrecognized GenerationBatch content in " + q + "; refusing to patch "
                 "(mlx-lm version mismatch?)")
    body = body.replace(old, new, 1)
text = head + body + tail
ast.parse(text)
open(q, "w").write(text)
print("patched:", q)
