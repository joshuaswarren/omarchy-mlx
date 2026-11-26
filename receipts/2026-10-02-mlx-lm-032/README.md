# mlx-lm 0.32 series gates (PR #30 / b6606be12) — MlxLm032 lane

Date: 2026-10-02. Runner: MlxLm032 (worker lane). Hosts: an M1 Max test
host (T6001, gpuwin windows, llm-inference stopped/restored around each)
and the x86 development box. Private venvs only; nothing system-wide was
touched. Notebook thread:
`apple-silicon-lab/entries/MlxLm032/20261002T211144Z-fleet-pr30-mlxlm032-ship-gates.md`.

PR #30 was merged before this lane's review completed (b6606be12); the
gates below ran against main as merged, per the orchestrator's rule:
keep if green, revert if any gate fails.

## Arms (identical wheel, only the mlx-lm line differs)

- OLD (shipping stack): `mlx_omarchy-0.32.4.dev202610021752+539d870e`
  (v0.7.19 release wheel, sha256 721020b035470f72…) + mlx-lm 0.31.3
  (PyPI) + the shipping patch series fetched at tag v0.7.19 (10 applied).
- NEW (merged series): same wheel + mlx-lm 0.32.0 (PyPI) +
  `patches/mlx-lm-0.32/` (10 applied) + rope-norm and qknorm patchers.
- Stamps asserted per the repo A/B rule: same `mx.__version__` both
  arms; `StopSequences` present only in NEW; `mlx_provenance.py` match
  on both; corpus `qwen38-2b-prompts.jsonl` sha256
  9299a3b2fc136a4c…; model Qwen3.8-2B-mlx-4Bit
  (`model.safetensors` b0d5de688567bf4a…).

## Review (static + suite)

- From-scratch apply onto pristine mlx-lm 94cdcae reproduces the PR
  receipt's patched-tree hash `e0623638535c7446` exactly (10/10 patches
  at fuzz 0, both patchers, no `.orig`/`.rej`). The PyPI mlx-lm 0.32.0
  wheel carries byte-identical patch-site files (the v0.32.0 release
  commit touches only `minimax_m3_vl.py`/`mistral4.py`/tests), and the
  series applies 10/10 on it as well — so the release can pin
  `mlx-lm==0.32.0` from PyPI with the same mechanism as 0.31.3.
- conv-ring refusal on 0.32 verified (exit 6 before touching the venv).
- Python suites: PR branch == pristine main (959 tests, 15 pre-existing
  environment errors in laya/bonsai2, 25 skipped). bun JS: 5/5 files
  pass.
- Known gap left by the PR: `install.sh` still pins mlx-lm 0.31.3 and
  its vendor fetch downloads only the flat `patches/` series, so a
  release-style install cannot serve a 0.32 venv (the apply script
  exits 5 with a clear message; verified). The pin swap (install.sh,
  requirements lock, installer fetch, docs) is a follow-up this lane
  owns; see "Pin swap" below.

## Gate: greedy-token identity vs the shipping stack (T6001)

`scripts/bench_decode.py` protocol (EOS suppressed, pinned length,
temp 0, seed 0, `MLX_DISABLE_COMPILE=1`), 477-token prompt, 5
alternating pairs per model, one boot:

| model | identity | decode tok/s old→new | prefill tok/s old→new |
|---|---|---|---|
| Qwen3.8-2B-4bit | **DIFFER** | 92.40 → 92.31 (−0.10%) | 1211.3 → 1210.0 (−0.10%) |
| Qwen3-4B-4bit | IDENTICAL (`2aec89056f9e7c82`) | 47.75 → 47.76 (+0.03%) | 504.5 → 506.7 (+0.45%) |
| Qwen3.5-9B-4bit | IDENTICAL (`28b786bb3864fabc`) | 27.11 → 27.13 (+0.06%) | 275.6 → 275.6 (+0.00%) |

Contract corpus (10 pinned prompts × 32 greedy tokens,
`qwen38-mlx-bench.py`, `ordered_records_sha256`):

- OLD `486872c410629f1d…`, NEW `3f6849fce9434929…` — differ on prompts
  4@19, 7@19, 8@5 (first differing token index; one token each).
- Bisect (window 2): with `MLX_OMARCHY_GDN_QKNORM_FUSE=0` and with
  `MLX_OMARCHY_NO_GREEDY_PRUNE=1` the NEW arm is byte-identical to
  itself — the qknorm epilogue and greedy prune are NOT the cause. A
  venv with the raw patch excluded (composed decode everywhere) differs
  from OLD on 2@24, 4@25, 7@19, 8@5 and from NEW on 2@24, 4@19.
- Reading: (a) the raw-vs-composed divergence is pre-existing and real
  on T6001 — the shipping stack's unconditional raw route perturbs 4/10
  contract prompts; the 0.32 series' conditional raw gate (skip raw
  when `lower_bound`/`allow_neg_eigval` are non-default) removes two of
  those four (prompt 8: NEW == composed exactly). (b) The residual
  old-vs-new flips persist with every fusion off, consistent with the
  0.31.3→0.32 model-math reassociation (upstream scaled the q/k-norm
  eps by `inv_scale**2`) flipping bf16 near-ties — not with a port
  defect. A/A control: OLD reproduced `486872c410629f1d` across runs.
- Coreglass's earlier equality (this series == shipping stack on
  T8103/T6021) ran TWO prompts at 64/128/256 tokens; the 10-prompt
  corpus is strictly more sensitive. No cross-chip digest gate was
  violated (per-host determinism holds; every arm reproduces itself).

## Quality gate (finding 1) — pre-declared threshold

Harness v2 (`scripts/bench/chat_model_bench.py`, GSM8K rows 0-19 at
openai/gsm8k@740312a, corrected IFE checks; the card held-out set is
NOT used), Qwen3.8-2B, OLD vs NEW, same boot, revision
0867d98bfb174b04…:

| | GSM8K | IFE |
|---|---|---|
| OLD (shipping 0.31.3) | 11/20 | 17/20 |
| NEW (0.32 series) | 12/20 | 17/20 |

Pre-declared threshold: no regression beyond noise (±1 item on 20).
Result: GSM8K +1 for NEW, IFE unchanged — **within noise, gate PASS**.
The numerics-policy condition for a greedy-token change is therefore
satisfied for the 2B near-tie flips recorded above. The fp32-reference
tiebreaker (which side of a flip is individually more accurate) was not
completed: the release wheel refuses CPU fallback by design and the
non-release CPU reference build was still running at receipt time;
recorded as an open follow-up, not a gate.

## Finding 2: stripped pacman package breaks provenance

Measured on the aarch64 release wheel: `mlx/lib/libmlx.so` is 25,302,096
bytes unstripped; `strip --strip-unneeded` → 22,096,568 (−3.2 MB);
`strip --strip-debug` → −3.9 KB. The venv tree is >1 GB, so keeping
symbols costs ~0.3% of the installed tree.

Decision: **option A — keep symbols.** `packaging/PKGBUILD.example`
gains `options=(!strip)` with the rationale in `package()`: the venv
ships byte-identical to the release wheel, so RECORD-based provenance
(`scripts/mlx_provenance.py`) holds on every packaged install without
new machinery. A build-manifest alternative (write post-strip hashes at
package time; check them in the provenance gate) was implemented first
and deleted in favor of A — A is simpler and guarantees identity with
the release WHEEL, not just a stripped derivative.
`tests/test_pkgbuild_contract.py` (2 tests) pins the contract: the
example declares `options=(!strip)` and never strips or rewrites
binaries. The real omarchy-pkgs recipe tracks the example (recipe not
in this repo).

## Verdict and state

- **KEEP b6606be12 (no revert).** Perf: no regression beyond noise on
  any model (worst −0.10%). Quality gate: PASS (12/20 vs 11/20 GSM8K,
  17/20 IFE — within the pre-declared ±1). Suites: no delta. Token
  identity: 4B/9B bit-identical; the 2B flips (3/10 contract prompts +
  the 477-token prompt) are the documented finding-1 class —
  pre-existing raw/composed split plus the upstream eps reassociation —
  permitted by the numerics policy once the quality gate passes, which
  it does.
- M2 cells (2B/4B/9B identity+perf A/B, oMLX re-measure): the 21:52Z
  mini-gap was canceled (FabricDsidAB owns the M2); run at 'M2 FREE' —
  this receipt gains a dated addendum with those numbers.
- Open follow-ups, in order: (1) M2 cells; (2) fp32 reference for the
  individual flips; (3) release pin cutover — install.sh
  `MLX_LM_VERSION`, requirements-lock.in `mlx-lm==0.32.0`, installer
  fetch of `patches/mlx-lm-0.32/` (today's vendor-layout fetch cannot
  serve a 0.32 venv), catalog/compatibility/serve docs, then the stale
  0.31.3 series deletion once the deployed 0.31.3 venvs are retired —
  sequenced AFTER the M2 cells confirm T6021, so Release0715 cuts with
  gates green on both remaining chips.
