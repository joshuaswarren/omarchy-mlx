# v0.7.32 gate summary (frozen wheel, 2026-10-10)

Freeze commit `ae5d0950bc4f2d47843c65112a212e21eb5e0060`. Every row below ran the frozen wheel bytes:

- `mlx_omarchy-0.32.4.dev202610101057+ae5d0950-cp314-cp314-linux_aarch64.whl`, sha256
  `a91fbe5ec2a8e4c197106d749739f696b2689f5926a97349d17e8827cc2f4712`.
- `omarchy-mlx-vendor-wheels-v0.7.32-cp314-aarch64.tar`, sha256
  `f0fcdf0fb0bd67f3898df338b460afcc9af9de1281e96c12c82164bca1539db0`.

Machines: 13-inch M1 (G13G), 16-inch M1 Max (G13C), 14-inch M2 Max (G14C). All three: kernel
7.1.12-2-12.6-sep-ARCH, Mesa 26.2.4-1, packaged Honeykrisp driver `omarchy-mlx-vulkan` 0.7.28-2 (reports
`Mesa 26.3.0-devel (git-e7631595df)`, library sha256 prefix `ac837b1c3aa08672`). The per-binary battery logs are
kept on the lab machines; this file carries the marker lines.

Rows replaced, not mixed in: the 16-inch M1 Max first ran on driver 0.7.22-1 and the 14-inch M2 Max on 0.7.28-1.
Both were upgraded to 0.7.28-2 during the gates and the rows rerun. The 14-inch M2 Max also wedged its GPU display
fence at about 17:35Z (a kernel worker blocked in `drm_atomic_helper_wait_for_fences`) and was rebooted at 19:15Z.
The M2 GPU rows below ran after that boot. The stamp row ran before it and reads asset hashes only.

## Rows

| Gate | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| g1 (fresh venv install, import, GPU op) | `G1_EXIT=0` (12:05Z) | `G1_EXIT=0` (17:02Z) | `G1_EXIT=0` (19:21Z, after the 19:15Z reboot) |
| stamp (runtime asset pin check) | `STAMP_EXIT=0` (12:19Z) | `STAMP_EXIT=0` (17:05Z) | `STAMP_EXIT=0` (19:10Z, reads asset hashes only) |
| g17 (15 patchers, mlx-lm 0.31.3 and 0.32.0) | `G17_DONE ok=4/4` (12:40Z rerun; the 12:13Z first run `ok=0/4` was a missing `patches/` directory) | `G17_DONE ok=4/4` (17:08Z) | `G17_DONE ok=4/4` (13:29Z, CPU gate) |
| g7d (Parakeet transcribe through the ANE worker) | `GATE7D_EXIT 0`, `GOLDEN_MATCH PASS` (12:00Z) | `GATE7D_EXIT 0`, `GOLDEN_MATCH PASS` (18:02Z) | not applicable |
| ANE legs g7c, g13, g15 | not part of the plan | g7c `GATE7C_EXIT 0`; g13 6 of 6; g15 4 of 4 (17:02Z) | not applicable |
| int8 matmul A/B and timing | `INT8_EXIT=0`, `ab-summary ALL MATCH` | `INT8_EXIT=0`, `ab-summary ALL MATCH` (log has no time header; last written 17:12Z, after the 16:56Z driver upgrade) | `INT8_EXIT=0`, `ab-summary ALL MATCH` (19:26Z) |
| fc2 repeat check | `REPEAT_SHA_MATCH` `5d2eb391db2925b6` (17:26Z) | `REPEAT_SHA_MATCH` `5d2eb391db2925b6` (17:35Z) | `REPEAT_SHA_MATCH` `5d2eb391db2925b6` (19:31Z) |
| standing battery, 45 suites | 41 green, 4 red | 42 green, 3 red | 42 green, 3 red |
| capability profiles (6 profiles, 7 cases) | 7 of 7 each | 7 of 7 each | 7 of 7 each |
| pinned decode (Qwen3-4B 4-bit, 64 tokens) | 22.04 tok/s median of 5 (v0.7.31: 22.08) | not run | not run |

The g7d transcript sha256 is `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` on both chips (the
golden pin), with `ane_mode = True` and `cpu_tensor_events = 0`.

On the 14-inch M2 Max, no ANE job ran during the int8 timing run (guard log: last ANE submit 18:49Z, next 19:33Z).
ANE jobs from another test did run beside battery parts 1 to 4 (submitted 19:33:38Z and 19:46:43Z). Those suites
check results, not speed. The battery count of 45 suites includes the ANE-linked runtime suite (14 of 14 cases).

## int8 matmul, same run, default route against scalar route (GMAC/s)

| Shape | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| qkv 6417x5376x16128, group 5376 | 343.8 vs 79.9 | 1367.1 vs 324.5 | 1829.2 vs 412.1 |
| fc1-swiglu 6417x5376x14336, group 5376 | 337.8 vs 80.1 | 1364.0 vs 324.0 | 1818.4 vs 411.9 |
| fc2 6417x14336x5376, group 1024 | 280.2 vs 79.1 | 740.8 vs 319.6 | 1846.2 vs 410.4 |

## Decode check (release step 6)

Both wheels were installed in fresh venvs, with the vendored dependencies from the vendor tar, and ran interleaved on
the 13-inch M1, five runs each, temperature 0, 64 pinned tokens, Qwen3-4B-Instruct-2507-4bit.

- Frozen wheel: 22.12, 22.18, 22.00, 21.96, 22.04 tok/s (median 22.04).
- v0.7.31 wheel: 22.09, 22.09, 21.99, 22.07, 22.08 tok/s (median 22.08).
- All ten runs generated the same 64 token ids (sha256 prefix `1c5bddb589b1edf8`).
- The wheel came from the staged release files, not the uploaded draft. The upload check below compares the
  uploaded bytes with the same sha256 values.
- An ANE job from another test was submitted on the machine three minutes before the run and may have overlapped part
  of it. Both wheels ran under the same conditions.

## Battery reds

All are defects in the tests. No product code is involved. Fix: #72.

- `gdn_legacy_policy`: a precondition went stale after #60.
- `conv_gemm_decomp`: a CPU reference stream is passed to a GPU-only call.
- `gdn_maskless_correctness`: a fixture path fixed at build time (6 of 6 pass with the fixtures staged).
- `sdpa_prefill_flash`: asserts the flash route without checking the device capability; one case also fails with an
  out-of-device-memory error on a 16 GB machine under load. #72 does not fix the out-of-memory case.

`fast_ops`: 47 of 47 cases passed; 8 assertions fail inside 3 cases marked as allowed to fail (sdpa vjp rep=1: 4,
gated-delta vjp equal heads: 1, gated-delta vjp GQA opt-in: 3). Identical on all three chips.
