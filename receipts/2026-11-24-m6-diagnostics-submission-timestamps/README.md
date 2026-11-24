# M6 diagnostics: per-submission GPU timestamps the board refuses (issue #526)

Date: 2026-11-24. Board: Apple M6 (G18G B1), Linux kernel
`7.1.12-aurora-m6usb-ARCH`, host `m6`. Driver: Honeykrisp (Mesa), Vulkan
ICD `/usr/share/vulkan/icd.d/asahi_icd.aarch64.json`.

Workload (all arms): Qwen3-4B (mlx-community/Qwen3-4B-Instruct-2507-4bit),
35-token prompt, 128 greedy tokens, single stream, the stock bench script
and the stock reference `stock-0.7.47-fixed/cap-timing-2/stock.json`
(`sha256 facee0db…`, inputs `sha256 9de5a270…`, both re-checked on every
arm).

## What the issue reported, and what is actually true

The issue reports a `VK_ERROR_DEVICE_LOST` in prefill on the diagnostics
wheel. It was never a device loss. The loss-free syscall trace (submit
ioctl id 29 only) shows three successful submits and then one submit
refused with `-EINVAL` in about 25 us, and the kernel logged

```
G18 compute metadata rejected: check d value 0
G18 compute admission rejected: invalid resource metadata
```

The second line is `admit_compute()` in
`drivers/gpu/drm/asahi/g18_drm.rs`: the G18 compute resource metadata
failed `G18ComputeMetadata::decode`, so the submit was refused with
EINVAL, and the driver reports a refused submit as a device loss. No GPU
timeout, no watchdog, no recovery ladder fired: a rejected submit, not a
hang.

The trigger is the diagnostics harness. The M6 has no per-submission
timestamp wheel at all (receipts/2026-11-13-g18g-batch-work: "This board
has no per-submission timestamp wheel, so the 2 to 6 ms submission band
used by the other rows was not measured"), but the Vulkan queue still
advertises timestamp support on this board (64 valid bits, 1 ns period),
and the GPU profiler trusted the advertisement. It wrote per-dispatch
timestamp commands and an execution barrier, so every submit carried a
timestamp-bearing compute command, and the board's admission check
refused it.

## Arms

| Arm | Wheel | Profiler | Result |
|---|---|---|---|
| A (original failure) | `+diag.db50a6464` | per-dispatch | `RuntimeError: [omarchy] dt.QueueSubmit(...) failed with VK_ERROR_DEVICE_LOST` in prefill; no ids |
| B (isolation) | `+diag.db50a6464` | off | completes; **ids identical to stock, 128/128**; no `profile.jsonl` (the harness was off) |
| C (fix) | `+diag.db50a6464` + the fix commit | per-dispatch | completes; **ids identical to stock, 128/128**; 2307 profile records, host costs only, no GPU ticks |

Arm C's meta record: `{"k":"meta","device":"Apple M6 (G18G B1)",
"period_ns":1.000000,"valid_bits":0,"pool":65536,
"label":"m6-qwen4b-35+128-fixed"}` - `valid_bits` 0 is the board gate, and
the stderr line is `[prof] board cannot carry per-submission timestamps;
host costs only`. Arm C's `ids.json` sha256 is
`11e1e188d4513a65f1feb4aa9cdc0d3ad212cd9c25e473f40565f7436c87e1e0`,
arm B's is `1ec65528f5d5c41b151c406de5a8d21a30fde846b3f9dd41fdbfa0df42655ecb`;
both arms ran in boot `7c3d6611-fe5a-4b70-b6f7-ba9d363e4956`.

Arm B is the run the issue asked for and did not have: the profiler off on
this exact diagnostics wheel. It completes with the stock ids, so the
wheel and the kernel are sound and the profiler's timestamp submits are
what the board refuses.

Arm C is the failing wheel's commit plus one commit (the board gate), so
the fix is the only variable between A and C.

## The fix

`CapabilityReport::submission_timestamps` (default true), set from
`submission_timestamps_supported(device_name)`; `kBoardsWithoutSubmissionTimestamps`
holds `G18G`. The GPU profiler reads it: on a board that cannot carry a
timestamp-bearing submit it creates no query pool and records no
timestamp or barrier command, so no submit carries a timestamp command
and the run continues with host costs only. The stderr line is
`[prof] board cannot carry per-submission timestamps; host costs only`.

Every other board keeps the queue's timestamp support; the doctest
"per-submission timestamps are gated per board" pins both sides.

## Provenance

| Item | Value |
|---|---|
| Arm A wheel | `mlx_omarchy-0.32.4.dev202611202136+diag.db50a6464d4fd3dcfbf07b7a8897e767c8da3eec-cp314-cp314-linux_aarch64.whl`, sha256 `90d3295dbb11f149d75c55fa5b95ec540c453c0e2ec71c4d342498f9f58cfaee` |
| Arm B | same wheel, profiler off (`MLX_OMARCHY_GPU_PROFILE` unset) |
| Arm C wheel | `mlx_omarchy-0.32.4.dev202611240041+diag.9225e68-cp314-cp314-linux_aarch64.whl` (fix commit `9225e68` on top of `db50a6464`) |
| Fix branch | `fix/526-g18g-submission-timestamps-main` (PR 793), base `origin/main` `197b068f3` |
| Scripts | `~/scratch/w7g-m6-profile35-20261120/profile35.sh` (A), `profile35-noprof.sh` (B), `profile35-fixed.sh` (C) |
| Boot id | recorded per arm in `capture*/boot_id` |

## Test

`overlay/tests/omarchy/test_runtime.cpp`: "per-submission timestamps are
gated per board" - `Apple M6 (G18G B1)` is refused; `Apple M1 Max
(G13C)`, `Apple M2 Max (G14C B1)`, `Apple M5 Max (G17C B1)`, `Apple A18
Pro (G17P B1)` and llvmpipe keep timestamp support.
