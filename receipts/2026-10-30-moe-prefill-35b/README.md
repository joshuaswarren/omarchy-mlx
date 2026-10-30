# Qwen3.6-35B-A3B prefill: stock mlx-lm reads the prompt 4.1x slower than mlx-lm with the omarchy patch series (2026-10-30)

Result: on a 16-inch M1 Max, `mlx-community/Qwen3.6-35B-A3B-4bit` prefills at 43.7 and 44.2 tokens per second with the venv's stock mlx-lm 0.31.3 and at 180.7 and 180.7 with the same mlx-lm after the omarchy serve patch series (`scripts/apply-mlx-lm-patches.sh`), same wheel, same machine, same prompt, interleaved runs. That is 4.1x. Per 512-token chunk the step takes 11.7 s against 2.88 s, and the number of GPU dispatches falls from 250,867 to 4,986. The expert path (`gather_qmm`) is not the cost. The stock mlx-lm runs the gated-delta update of the 30 linear-attention layers as a Python loop of tiny operations, one token at a time; the patch series routes it to `mx.fast.gated_delta_update`.

The 13 tokens per second that the 14-inch M2 Max reported for this model in earlier runs came from a venv built by hand (`main-venv-fresh`), and I could not read that venv (the machine was down). My reading is that it skipped the patch step; I have not confirmed it. The M2 must be re-measured with a patched venv.

## What ran

- Machine: 16-inch M1 Max, kernel `7.1.12-2-20261029.2-lab-ARCH`, idle, one process per run.
- Wheel: diagnostics build of `42b30a2d` (main after #303; the commits since touch no backend code), `mlx-lm` 0.31.3, `Qwen3.6-35B-A3B-4bit` from the local Hugging Face cache (snapshot revision in the first log), `MLX_DISABLE_COMPILE=1`.
- Prompt: the repository's own docs as real English text; a 512-token warmup chunk, then 2048 tokens in chunks of 512 (`--prefill-step-size 512` as in the earlier runs). One bracketed window per chunk (`data/prefill_profile_driver.py`).
- U0 is the venv's own mlx-lm (no `.omarchy-serve-patches`, no `mx.fast.gated_delta_update` in `models/gated_delta.py`). F is a copy of pristine 0.31.3 with the series applied by `apply-mlx-lm-patches.sh` of `42b30a2d` (pure Python, put first on `PYTHONPATH`).

| run | prefill tokens per second | step per 512-token chunk (median of the last 3) |
| --- | --- | --- |
| U0, round 1 | 43.67 | 11.75 s |
| F, round 1 | 180.68 | 2.876 s |
| U0, round 2 | 44.18 | 11.65 s |
| F, round 2 | 180.66 | 2.877 s |

Earlier unprofiled run, venv's own mlx-lm: 44.34 tokens per second; with the sorted-expert tile route off (`MLX_OMARCHY_GATHER_QMM_TILE=0`): 36.40, so the tile route is taken and worth 18 percent.

## Where the unpatched time goes (GPU profile, `data/first-profile-run.log`)

Per 512-token chunk: 250,867 dispatches, of which `ElementwiseLiteF32` 107,640, `CastBF16F32` 77,311, `ReduceF32` 30,720, `CopyGeneralBF16` 15,680, `CastF32BF16` 15,390. Over the five chunks of the run each of the first three small shapes appears 307,200 times for `CastBF16F32` (n = 4096), and the same for `ElementwiseLiteF32` at n = 524288 (the 32x128x128 recurrent state): 512 tokens x 30 linear-attention layers x 4 x 5 chunks. The mixture-of-experts block is about 60 dispatches per layer and uses `GatherQmmTileBF16` (5 ms of GPU time in the whole profile). After the series: 4,986 dispatches per chunk.

## What this does not show

- The GPU timestamps of a prefill chunk are small (3.2 s of 19.5 s unpatched, 107 ms of 3.07 s patched) and the 2026-10-03 prefill receipt calls them unusable on this Mesa line, so I do not claim where the remaining 2.9 s of the patched chunk go, nor that 181 tokens per second is a limit.
- Not compared with llama.cpp on this chip. llama.cpp reads 293 tokens per second on a 14-inch M2 Max (a different chip), so 181 here is not a comparison with it.
- The M2 itself, and the 13 tokens per second, were not reproduced.
- Decode was not measured.

## The change in this pull request

- `honeykrisp_identity.h`: `mlx_lm_patch_warning(lib_dir)` reads `<site-packages>/mlx_lm/models/gated_delta.py` beside the loaded libmlx and warns when it has no `mx.fast.gated_delta_update`; `note_mlx_lm_patches` prints it once per process, at the same point as the stock-Mesa warning (`device.cpp`). `MLX_OMARCHY_NO_MLXLM_WARNING=1` silences it. The backend cannot see the unpatched loop as a fallback (it never reaches the gated-delta primitive), so the check reads the file. A doctest was written first and failed to compile before the function existed (`overlay/tests/omarchy/ane/test_worker_gate.cpp`); the standalone build of that case passes (8 assertions). `device.cpp` compiles with the call; I did not run the warning on a device.
- `docs/INSTALL.md`: the manual install block now ends with the `--patch-venv` line instead of leaving it to a later paragraph. `docs/kernel-flags.md` has the new switch.

## Files

`PREREG.md` (the plan and two amendments, each written before the step it precedes; Amendment 2 corrects a dispatch count I reported wrongly); `data/`: run logs, per-run summaries, the driver and ticket scripts (host names replaced by `hostB`).
