# v0.7.32 gate-run RUNBOOK (generated 2026-10-08 from the v0.7.31 receipts:
# receipts/2026-10-07-release-0.7.31/gates-run/*). Goal: any host window is
# minutes from firing. Nothing here runs until Main says the wheel exists.

## 0. Inputs to stage BEFORE any window (per host, as FILL)
- Draft assets for v0.7.32 into `~/v0732-assets/` on the target host:
  wheel `mlx_omarchy-0.32.4.dev*-cp314-cp314-linux_aarch64.whl`,
  `omarchy-mlx-vendor-wheels-v0.7.32-cp314-aarch64.tar` (+ `.sha256`),
  `SHA256SUMS`. Verify on arrival: `cd ~/v0732-assets && sha256sum -c SHA256SUMS`.
- Gate scripts from main tip:
  `rsync -a <worktree>/scripts/release-gates/ <host>:~/v0732-gates/release-gates/`
  plus the standalone probes (they live in receipts/, not scripts/):
  `g16-qmm-batch.py`, `g16b-qmm-route-probe.py`, `g-g13c.sh`,
  `g-hold-g13g.sh`, `g17-patch-series.py`, `apply-mlx-lm-patches.sh`,
  `patches/` (both series), `70-omarchy-mlx-cpu-dma-latency.rules`
  (source: receipts/2026-10-06-release-0.7.29/gates/ — unchanged since).
- mlx-lm pristine wheels for g17: `pip download mlx-lm==0.31.3 mlx-lm==0.32.0 --no-deps -d ~/v0732-dl/`.

## 1. Env contract (placeholders to fill per host — from scripts/release-gates/env.sh)
- `TAG=v0.7.32` (or the draft tag Main cuts)
- `ASSETS_DIR=$HOME/v0732-assets`
- `GATE_ROOT=$HOME` (state lands in `~/v0.7.32-gate-home`, logs `~/v0.7.32-gate-logs`)
- `TAG_SHA=<40-char sha of the tag commit>` (REQUIRED: gate_ensure_install_tree asserts it)
- `EXPECTED_WHEEL_SHA256=<sha>` / `EXPECTED_VTAR_SHA256=<sha>` (REFUSE on mismatch)
- M2 only: `GPU_LOCK=/tmp/m2-gpu.lock`; jw16 legs: `RUN_JW16=1 JW16_SSH=16m1mbp SERVING_VENV=~/v0731-venv` (fresh path per host)
- g16/g16b: `VK_DRIVER_FILES` leg B overrides to the coreglass ICD; leg P uses the packaged ICD (no var).
- g-g13c: runs INSIDE a gpu-turn ticket (`GPU_TURN_TICKET=1` from the driver); expects `EXPECTED_WHEEL_SHA256` set.
- g-hold: `HOLD_WHEEL=~/v0732-assets/<wheel> HOLD_VENDOR_TAR=~/v0732-assets/<vtar> HOLD_RULE_SRC=~/v0732-gates/70-omarchy-mlx-cpu-dma-latency.rules` (HOLD_MODEL defaults /var/tmp/MesaParity/model); udev expectation root:video 0660.

## 2. Gate order and measured durations (v0.7.31 evidence, M2 12.2)
From gates.done (receipts/2026-10-07-release-0.7.31/gates-run/m2-gates-done-final.log):
| gate | wall (v0.7.31 measured) |
|---|---|
| g1-clean-install | 43 s |
| g2-online-9b | 203 s |
| g3-online-4b-card | 62 s |
| g4-offline | 16 s |
| g5-laya | 0 s |
| g6-codec | 11 s |
| g7a-packaged-icd | 0 s |
| g7b-system-install | 17 s (after INSTALL_TREE auto-staged; driver now does this — 97d31bbf0) |
| g8-kokoro | 10 s |
| g9-speak-queue | 12 s |
| g10-kokoro-primer | 123 s |
| g11-card-9b | 21 s |
| g12-kokoro-stream | 89 s |
| g14-routing | 23 s |
| M2 total | ~10.5 min |

jw16 ANE legs (from jw16-gates-receipt-jw16.log, v0.7.31):
build-wheel 169 s, g13-build 258 s, g15-build 8 s, g7c 148 s, g7d 43 s,
g13-run 135 s, g15-run 1 s → ~13 min + the g7c/g7d windows.

Standalone probes (not in run-all):
- g16 legs P+B: ~2-3 min each (12/12 rows bit-identical check)
- g16b legs P+B: ~16 s each
- g-g13c legs P+B: ~2-5 min each (inside gpu-turn tickets)
- g-hold legs P/B: 81 s / 110 s (v0.7.31 measured on jwm1)
- g17 patch-series: ~2 min (dev box measured 4/4 in 2.1 s per apply pair)

## 3. Exact commands per host

### M2 (G14C) — full battery + Bonsai leg
```
ssh <M2_SSH>
export TAG=v0.7.32 ASSETS_DIR=$HOME/v0732-assets GATE_ROOT=$HOME
export TAG_SHA=<tagsha> EXPECTED_WHEEL_SHA256=<wheelsha> EXPECTED_VTAR_SHA256=<vtarsha>
bash ~/v0732-gates/release-gates/run-m2-battery.sh   # wraps run-all.sh; PASS rule: 14 LOCAL gates RC=0, jw16 gates SKIPPED-only
# then the Bonsai G14C leg ticket via ~/bin/gpu-turn (per its queue entry)
```

### jwm1 (G13G) — main battery + probes
```
ssh jwm1
export TAG=v0.7.32 ASSETS_DIR=$HOME/v0732-assets GATE_ROOT=$HOME
export TAG_SHA=<tagsha> EXPECTED_WHEEL_SHA256=<wheelsha>
bash ~/v0732-gates/release-gates/g1-clean-install.sh
for g in g2-online-9b g3-online-4b-card g4-offline g5-laya g6-codec g7a-packaged-icd g7b-system-install g8-kokoro g9-speak-queue g10-kokoro-primer g11-card-9b g12-kokoro-stream g14-routing; do bash ~/v0732-gates/release-gates/$g.sh || echo "FAIL $g"; done
python3 ~/v0732-gates/g16-qmm-batch.py            # leg P (packaged ICD)
VK_DRIVER_FILES=<coreglass-icd> python3 ~/v0732-gates/g16-qmm-batch.py   # leg B
python3 ~/v0732-gates/g16b-qmm-route-probe.py     # legs P+B same pattern
bash ~/v0732-gates/g-hold-g13g.sh                  # with HOLD_* env above
python3 ~/v0732-gates/g17-patch-series.py          # or the jwm1-g17-run2.sh pattern (repo apply script, layout 2)
```

### jw16 (G13C) — restage first (home wiped), then battery
```
ssh 16m1mbp   # plain alias, ControlMaster (NEVER ControlPath=none)
mkdir -p ~/v0732-assets ~/v0732-gates
# stage assets + scripts as FILL (section 0), sha256sum -c, then:
export TAG=v0.7.32 ASSETS_DIR=$HOME/v0732-assets GATE_ROOT=$HOME
export TAG_SHA=<tagsha> EXPECTED_WHEEL_SHA256=<wheelsha> SERVING_VENV=$HOME/v0732-venv
# ANE legs via jw16-gates.sh (stages to /tmp, runs g7c/g7d/g13/g15)
bash ~/v0732-gates/jw16-gates.sh
# g-g13c legs P+B inside gpu-turn tickets (GPU_TURN_TICKET=1 contract):
bash ~/v0732-gates/g-g13c.sh    # leg P; repeat with VK_DRIVER_FILES override for leg B
```

### dev box (CPU) — g17 only
```
bash /tmp/g17-0732-run.sh   # pattern: pristine 0.31.3/0.32.0 wheels + main's apply script; expect ok=4/4
```

## 4. Standing battery: exact binary list per chip (v0.7.32 additions)

Build (main tip, static test configure) then run with ctest or directly.
HEADLINE-fix binaries (MUST run on each chip):
- omarchy_int8_matmul_tests (int8 rows*n > 16,776,960)
- omarchy_copy_offset_tests (CastBool > 67,107,840)
- omarchy_fast_ops_tests (fused rope_rms_norm >= 65,535 rows)
- omarchy_linalg_ops_tests + omarchy_eig_ops_tests (batch > 65,535)
- omarchy_indexing_ops_tests (bool MaskedScatter)
- omarchy_reduce_ops_tests (ArgReduce NaN)

Full AGENTS.md standing battery + additions (run on EACH chip):
omarchy_runtime_tests omarchy_primitive_tests omarchy_matmul_family_tests
omarchy_fast_ops_tests omarchy_kv_ops_tests omarchy_indexing_ops_tests
omarchy_reduce_ops_tests omarchy_shape_ops_tests omarchy_linalg_ops_tests
omarchy_copy_offset_tests omarchy_distributed_tests
omarchy_compiled_tape_tests omarchy_fft_ops_tests omarchy_fft_general_tests
omarchy_eig_ops_tests omarchy_take_fill_tests omarchy_take_bool_tests
omarchy_conv_tests omarchy_complex_ops_tests omarchy_select_layout_tests
omarchy_fast_regression_tests omarchy_scatter_determinism_tests
omarchy_eq_math_tests omarchy_fused_chain_tests omarchy_error_contract_tests
omarchy_int8_matmul_tests omarchy_trig_reduction_tests
omarchy_capability_sim_tests omarchy_qmv_batch_tests
omarchy_gdn_maskless_correctness_tests omarchy_gdn_decode_batch_tests
omarchy_gdn_fast_route_repeat_tests omarchy_gdn_legacy_policy_tests
omarchy_gdn_prefill_profile_tests omarchy_sdpa_causal_ragged_tests
omarchy_conv_gemm_decomp_tests omarchy_ane_bundle_tests
(+ the sdpa norm/prefill/decode binaries per CMake target names).

One-liner (build dir): `ctest --output-on-failure` (ctest selects the
omarchy_* tests registered via add_test).

Chip status: G13C jw16 DONE (c57d5ea67, DispatchClamp); G13G jwm1 OFFLINE
(macOS boot experiment, jwm1-parity lane — ASK MAIN); G14C M2 via
idle-guard. FREEZE waits on: TensorFold full-depth rerun + all three
batteries green from main tip.

## 4. Standing battery: exact binary list per chip (v0.7.32 additions)

Build (main tip, static test configure) then run with ctest or directly.
HEADLINE-fix binaries (MUST run on each chip):
- omarchy_int8_matmul_tests (int8 rows*n > 16,776,960)
- omarchy_copy_offset_tests (CastBool > 67,107,840)
- omarchy_fast_ops_tests (fused rope_rms_norm >= 65,535 rows)
- omarchy_linalg_ops_tests + omarchy_eig_ops_tests (batch > 65,535)
- omarchy_indexing_ops_tests (bool MaskedScatter)
- omarchy_reduce_ops_tests (ArgReduce NaN)

Full AGENTS.md standing battery + additions (run on EACH chip):
omarchy_runtime_tests omarchy_primitive_tests omarchy_matmul_family_tests
omarchy_fast_ops_tests omarchy_kv_ops_tests omarchy_indexing_ops_tests
omarchy_reduce_ops_tests omarchy_shape_ops_tests omarchy_linalg_ops_tests
omarchy_copy_offset_tests omarchy_distributed_tests
omarchy_compiled_tape_tests omarchy_fft_ops_tests omarchy_fft_general_tests
omarchy_eig_ops_tests omarchy_take_fill_tests omarchy_take_bool_tests
omarchy_conv_tests omarchy_complex_ops_tests omarchy_select_layout_tests
omarchy_fast_regression_tests omarchy_scatter_determinism_tests
omarchy_eq_math_tests omarchy_fused_chain_tests omarchy_error_contract_tests
omarchy_int8_matmul_tests omarchy_trig_reduction_tests
omarchy_capability_sim_tests omarchy_qmv_batch_tests
omarchy_gdn_maskless_correctness_tests omarchy_gdn_decode_batch_tests
omarchy_gdn_fast_route_repeat_tests omarchy_gdn_legacy_policy_tests
omarchy_gdn_prefill_profile_tests omarchy_sdpa_causal_ragged_tests
omarchy_conv_gemm_decomp_tests omarchy_ane_bundle_tests
(+ the sdpa norm/prefill/decode binaries per CMake target names).

One-liner (build dir): `ctest --output-on-failure` (ctest selects the
omarchy_* tests registered via add_test).

Chip status: G13C jw16 DONE (c57d5ea67, DispatchClamp); G13G jwm1 OFFLINE
(macOS boot experiment, jwm1-parity lane — ASK MAIN); G14C M2 via
idle-guard. FREEZE waits on: TensorFold full-depth rerun + all three
batteries green from main tip.

## 5. Receipt rules (unchanged from v0.7.31)

- Every gate logs `WHEEL_IDENTITY` + `uname_r` (feb57d020, 2811cf901).
- SUMMARY generated from logs by script, never hand-edited.
- Land under receipts/2026-10-08-release-0.7.32/gates-run/ via fetch+rebase.
- scrub /home/<user>/ paths before commit (privacy-check blocks them).

## 6. v0.7.32 FREEZE gate plan (per Main, 2026-10-09; CPU prep only until FREEZE)

Hosts and chip roles: M2 = G14C, jw16 = G13C, jwm1 = G13G. Kernel of record
in every receipt: `uname -r` AND `pacman -Q mesa` (hosts are on Mesa
26.2.4 now — the StockWarn threshold; any host below 26.2.4 warns by
design).

Order on EVERY host:
1. g1 fresh-home install of the draft wheel (drivers: g1 then g17 first —
   a wheel that cannot install or patch is dead on arrival).
2. g17 patch-series, both mlx-lm lines rc=0 + idempotent.
3. Standing battery (the section-4 binary list) on the GPU.
4. int8 headline check per chip (the clip shapes through the wheel venv:
   qkv 6417x5376x16128 g=5376, fc1 swiglu 6417x5376x14336 g=5376, fc2
   6417x14336x5376 g=1024; bit-exact A/B rows<=128 vs naive; GMAC/s with
   the wheel stamp; fc2 expected ~734 GMAC/s class, coop arms are the
   headline).
5. TensorFold guide gate: render/follow check that the guide's expected
   seed-1 mp4 sha256 36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07
   reproduces byte-identical on the M2 (per docs/tensorfold-video.md; the
   guide documents the hash holding across wheels 53bc1e3 and 95e7b6f on
   one chip).
6. PR 50 stamp gate on the built wheel: verify_runtime_assets passes
   (pre-stamp wheels FAIL by design — the wheel must carry the stamped
   ANE worker; this is a positive check on the stamp, not a skip).

Ticket minutes per host (expected, for scheduling):
- M2 (G14C): g1 ~1, g17 ~3, battery ~25-35 (49 binaries; the M2 runs them
  fastest), int8 bench ~6 (three shapes x two routes x 3 submits),
  TensorFold render gate ~60 (58:44 measured + margin) or the sha
  re-check only ~2 if Main accepts the receipt-replay form, PR 50 stamp
  check ~1. TOTAL ~40-50 min without the video render; ~100-110 with.
- jwm1 (G13G): g1 ~1, g17 ~3, battery — the 15-suite run measured
  10:33-11:23Z (~50 min wall for all 15 at MAXMIN 20 each; the binaries
  themselves sum to ~35 min), int8 bench ~6, stamp check ~1. TOTAL ~45-60
  min.
- jw16 (G13C): g1 ~1, g17 ~3, battery ~35-45, int8 bench ~6 (this is the
  headline chip for the coop multipliers), stamp check ~1, plus the
  jw16-only ANE legs (g7c ~3, g7d ~1, g13 build+run ~7, g15 ~1). TOTAL
  ~55-70 min.

Rules carried over: no submit > 20 s per call; lane 'Release0729'; one
flock-protected append; receipts carry WHEEL_IDENTITY + uname -r + Mesa
version; SUMMARY generated from logs; land via fetch+rebase.
