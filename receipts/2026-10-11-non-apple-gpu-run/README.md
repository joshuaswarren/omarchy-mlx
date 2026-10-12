# Non-Apple GPU development run: RTX 5060 Ti and Intel UHD 770

Record for the non-Apple rows in
[docs/compatibility.md](../../docs/compatibility.md#non-apple-gpus-development-drivers)
and the [Other Vulkan GPUs](../../README.md#other-vulkan-gpus-development) table.
Source of record: [PR #111](https://github.com/joshuaswarren/omarchy-mlx/pull/111)
(merged 2026-10-11, merge commit `7b58217a`). The runs are the PR author's, on
the author's host; this file transcribes that record so the numbers in the docs
are auditable from the repo. No dispatch trace dump was captured; dispatch
behavior is recorded as observed output below.

## Host, drivers, source commit

- Host: Linux 7.2.9 (CachyOS), one box holding an NVIDIA RTX 5060 Ti (discrete)
  and an Intel UHD 770 (integrated).
- NVIDIA proprietary driver 615.71.09; Mesa ANV 26.2.4; Vulkan loader 1.4.363.
- Source: `main` at `e95e86b6` (MLX 0.32.3, `9c3d355`) plus the PR #111 branch.
- Build: `cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON`.

## Exact command

```
MLX_OMARCHY_ALLOW_NON_APPLE=1 \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/<nvidia|intel>_icd.json
```

One driver per run.

## Numerical results

- `omarchy_runtime_tests`, Intel UHD 770: 55/55 passed.
- `omarchy_runtime_tests`, RTX 5060 Ti: 54/55 passed. The miss is
  "independent streams measure against the serialized sum": ratio 0.78 to 0.88
  across 5 runs against the 0.95 floor. The PR changes nothing that check
  measures; without it the backend cannot start on NVIDIA, and NVIDIA's fixed
  subgroup size of 32 makes the subgroup pin a no-op there.
- Op check, GPU stream against CPU stream: add, sigmoid, softmax, copy, argmax,
  rms_norm, rope, sdpa, take, bf16 matmul, 4-bit qmm at M 1 to 24, K 2048,
  N 1024 to 6144. Intel: 0 bad, 0 failed. RTX 5060 Ti: within bf16 tolerance.
  The earlier timeline timeout did not reproduce; a rerun of all qmm4 cases
  through N 6144 passed.
- End to end through mlx-serve (linked from PR #111) on the
  5060 Ti: Qwen3-1.7B 4-bit decodes at about 216 tok/s on a short prompt.

## Dispatch behavior and recovery

- Pipelines require the reported `subgroup_size` when the driver reports a
  range and supports `subgroupSizeControl`. Intel reported 8 to 32 and ran
  small workgroups at 16 lanes, which gave wrong subgroup results before the
  pin. NVIDIA reports a fixed 32.
- Discrete GPUs sort ahead of integrated ones; the Intel iGPU enumerated ahead
  of the NVIDIA card before the change.
- The NVIDIA run prints one `SUBMIT-RECOVER ... resubmitted 1 stalled batch`
  line per run; the submit-recovery ladder fired once per run. No device
  reopen or reboot was needed.

## Not run

- Apple machines; lavapipe (not installed on the host).
- AMD: no qualifying run.

## Known limits recorded with the run

- Every array lives in host-visible memory. On the discrete card that is VRAM
  behind ReBAR, so CPU reads over PCIe measured about 3 MB/s. GPU-side work is
  unaffected.
- From mlx-serve on NVIDIA, cause not established: decode fell from 216 to 50
  tok/s as context grew to 2.4K tokens; a multi-token-prediction draft head
  accepted 3% of predictions on Vulkan against 40% on CUDA.
