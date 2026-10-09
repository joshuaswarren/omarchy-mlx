# Allocator: geometric size bins, and the 16 GB offload out-of-memory they fix

Claim: since commit `b20211a0b` every allocation above 1 MiB is rounded to the next power of two, which wastes up to 100% per buffer. On a 16 GB M1 this made the first prefill chunk of oMLX expert offload (Qwen3-30B-A3B 4-bit, 0.25 residency) fail with `VK_ERROR_OUT_OF_DEVICE_MEMORY` at a 32 MiB request while 13 GB of RAM was free. Rounding to 8 sizes per power of two (at most 12.5% waste) fixes it, and cuts device memory of the stock model on a large host by a quarter.

## Source commits

| item | commit |
|---|---|
| first bad commit | `b20211a0b` omarchy allocator: bin variable-sized buffers (2026-10-05) |
| its parent, last good in the allocator series | `ae8b8018b` |
| base for the speed A/B (parent of the fix) | `268a3322` |
| fix | `05606841f` (wheels were built from this commit); `c62ed6d88` on top changes two comments and docs only |

## Host and device

| | M1 (G13G), 16 GB | M1 Max (G13C), 62 GB |
|---|---|---|
| kernel | 7.1.12-2-12.6-sep-ARCH | same |
| Mesa, Vulkan driver | 26.2.4 (system), Honeykrisp, Vulkan 1.4.354, conformance 1.4.0.0 | 26.2.4 (system) |
| device | Apple M1 (G13G B1), integrated GPU | Apple M1 Max (G13C) |
| heap the backend reports to oMLX (`max_recommended_working_set_size`) | 8,111,783,936 bytes, half of RAM | not captured |

`vulkaninfo` device lines were captured on the M1 only. Boot firmware identity, read from the device tree after the runs:

| | M1 Max | M1 |
|---|---|---|
| board | apple,j316c apple,t6001 apple,arm-platform | apple,j293 apple,t8103 apple,arm-platform |
| iBoot stage 1 | mBoot-18000.161.10 | mBoot-20457.1.29 |
| iBoot stage 2 | iBoot-8422.141.2 | iBoot-8422.141.2 |
| m1n1 stage 1 | v1.6.1-dirty | v1.6.1-dirty |
| m1n1 stage 2 | v1.6.1-omarchy.aurora14 | v1.6.1-omarchy.aurora14 |
| OS firmware | 13.5 | 13.5 |
| system firmware | 26.6.2 | 27.0 |

## Model

Qwen3-30B-A3B-Instruct-2507, MLX 4-bit, group size 64 (router gate 8-bit). Repository snapshot `e9675aa3ca5f900ccef55267914466d55ab325fa`. File sha256:

```
70919a0f0b7d86c3100e30dfc2c72eb29909b841f9414b9c5ae3e8673ec9ff8c  model-00001-of-00004.safetensors
1a0387973b19b0bac1201358d1f56989deaf595da34f307ec29777fa6f8469b8  model-00002-of-00004.safetensors
d1a627729b2791dba7db5cda5dd0d24ea4c44c5131981c258598d5d0c80e0df3  model-00003-of-00004.safetensors
b02f0e9f626bd2f5bb41e057216c9d6f594ddeac91cc3f2161cc935d050a12c6  model-00004-of-00004.safetensors
c100efb419233c6be585014ebd9f5a8c2a4fe9de63be6271172f0f70f1c315ea  model.safetensors.index.json
e1caf81b25b71686af871f1147290a81cbcb81283b638f98957293990e031417  config.json
```

## Commands

Bisect and fix check (M1 16 GB). One venv with the oMLX dependencies, only the mlx_omarchy wheel swapped between runs. Each run holds the GPU lock through the lab's ticket wrapper (`gpu-turn`, at most 12 minutes):

```
OMLX_MOE_OFFLOAD_LOOKAHEAD=0 W7G_RESIDENCY=0.25 W7G_PREFILL=16 W7G_CHUNK=16 \
OMLXPERF_OMLX_PATH=<oMLX v0.7.0 tree with the Linux patch series> \
<venv>/bin/python w7g-offload-mem.py
```

The probe loads the model lazily, wraps the experts with `apply_moe_expert_offload(model, path, 0.25)`, calls `materialize_offload_state`, then runs the first prefill chunk (16 tokens), and prints `mx.get_active_memory()`, `get_cache_memory()` and `get_peak_memory()` after each stage. It prints the same line when the chunk fails.

Full workload on the fix wheel (same probe family as the oMLX pin receipt): 96 prefill tokens in chunks of 16, 32 greedy decode tokens, oMLX tree at the new pin.

Growing-KV speed A/B (M1 Max): stock mlx_lm, ten prompt lengths 64 to 1000 that are not powers of two, each prefilled in 128-token chunks then 20 decode tokens; then one 256-token prompt and 300 decode tokens. Arm order base, fix, fix, base, same venv, only the wheel swapped.

## Result 1: where the regression starts (M1 16 GB)

First prefill chunk, per wheel. "Distance" is the commit count from wheel `58724762`.

| wheel (mlx_omarchy dev tag) | distance | after materialize | first chunk |
|---|---|---|---|
| 202610031525+58724762 (the lab's older oMLX venv, see Provenance) | 0 | 4077 MB | pass |
| 202610050725+5c15fba | 211 | 4077 MB | pass |
| 202610090448+ae8b801 (parent of the binning commit) | 319 | 4077 MB | pass |
| 202610090452+b20211a (the binning commit) | 320 | 5436 MB | out of memory, 32 MiB request; active 5760 MB, cache 1858 MB at the failure |
| 202610061037+2540b10, 202610061041+bf62cfb, 202610071056+ef70b8cc, 202610071347+9b5c938, 202610081002+145886c | 364 to 660 | 5436 MB | out of memory |
| 202610090456+560684 (the fix) | | 4077 MB | pass; active 5008 MB, cache 2414 MB after the chunk |

The numbers follow from the checkpoint headers: with 32 slots per layer the 432 expert slot arrays total 4077 MB exact and 5436 MB with power-of-two bins (a 24 MiB array becomes 32 MiB). The capacity of 32 slots is the one the backend run used.

## Result 2: full offload workload on the fix wheel (M1 16 GB)

Prefill of 96 tokens 51.6 s, decode 851 ms per token, greedy token sha `b530b093661caf6f`. The same sha comes from every other run of this workload on a subgroup-kernel wheel (an allocator-drain-only wheel: 53 to 54 s and 855 to 882 ms).

## Result 3: no speed cost, large saving (M1 Max, stock model, base vs fix, arm order base, fix, fix, base)

| | base | fix |
|---|---|---|
| phase A wall (ten drifting prompt lengths) | 176.9 s and 176.8 s | 177.0 s and 176.8 s |
| phase B wall (300 decode tokens, growing KV) | 40.1 s and 40.0 s | 40.1 s and 40.0 s |
| decode ms per token, phase A median | 96.4 and 95.8 | 95.3 and 94.4 |
| decode ms per token, phase B median | 97.6 and 97.4 | 97.3 and 96.9 |
| device memory at the end, active | 22,871 MB | 17,208 MB |
| device memory at the end, peak | 23,124 MB | 17,458 MB |
| device cache at the end | 1,228 MB | 1,510 MB |
| greedy token sha | `a5b0c7f66acd660d` in all four arms | |

The fix lowers device memory by 5.6 GB (24.8%) for the stock model and leaves speed unchanged within the run-to-run spread.

## Provenance

mlx_provenance.py per measured binary (verified = on-disk mlx.core and libmlx.so hashes equal the installed wheel RECORD):

| chip | wheel or venv | verified | dist version | version matches mx | mlx.core sha256 | libmlx.so sha256 |
|---|---|---|---|---|---|---|
| M1 Max | 202610090526+268a332 | match | 0.32.4.dev202610090526+268a332 | yes | f0ebfdafc986 | 664e479e9831 |
| M1 Max | 202610090456+560684 | match | 0.32.4.dev202610090456+560684 | yes | f0ebfdafc986 | a091e52d8e78 |
| M1 | 202610090448+ae8b801 | match | 0.32.4.dev202610090448+ae8b801 | yes | dcd72c640d75 | 8e1724687a21 |
| M1 | 202610090452+b20211a | match | 0.32.4.dev202610090452+b20211a | yes | dcd72c640d75 | 254bf8d7e944 |
| M1 | 202610090456+560684 | match | 0.32.4.dev202610090456+560684 | yes | f0ebfdafc986 | a091e52d8e78 |
| M1 | omlx-perf | mismatch | 0.32.4.dev202610031525+58724762 | yes | 6715a0f96c3f | 4160e6a5b74d |
| M1 | venv | match | 0.32.4.dev202610050725+5c15fba | yes | dcd72c640d75 | b12e98579349 |
| M1 | venv-ab-2540b10 | match | 0.32.4.dev202610061037+2540b10 | yes | dcd72c640d75 | 56969c2f067f |
| M1 | venv-ab-bf62cfb | match | 0.32.4.dev202610061041+bf62cfb | yes | dcd72c640d75 | c49388e02bb6 |
| M1 | venv-h13main | match | 0.32.4.dev202610071056+ef70b8cc | yes | dcd72c640d75 | a8f1116df9ed |
| M1 | venv-release | match | 0.32.4.dev202610071347+9b5c938 | yes | dcd72c640d75 | 4cec6a66660c |
| M1 | venv-main | match | 0.32.4.dev202610081002+145886c | yes | f0ebfdafc986 | 982be9ea576b |

All wheels built for this work (`ae8b801`, `b20211a`, `268a332`, `560684`) and the other venvs verify against their wheel RECORD. One exception: the older oMLX venv labelled `58724762` fails the check. Its `libmlx.so` on disk (4160e6a5b74d) is not the file its wheel RECORD lists (b58cae8f7bee), while `mlx.core` matches and the versions agree. The cause is not known. Its row in Result 1 therefore describes that venv's binaries, not wheel `58724762`. The bisect does not depend on it: the wheels built from `ae8b8018b` (pass) and `b20211a0b` (fail) both verify, and so does `5c15fba` (pass). The `mlx.core` extension is byte-identical across `ae8b801`, `b20211a` and `5c15fba` to `9b5c938`, and across `268a332`, `560684` and `145886c`; only `libmlx.so` differs between the base and the fix.

## Dispatch trace

Not captured. The backend has no dispatch-list trace to capture, and the change touches buffer sizes only. Unchanged kernel selection is shown indirectly: greedy token ids are identical in all four base and fix arms of Result 3, and the full workload on the fix wheel gives the same token sha as every reference run.

## Timing and thermal procedure

Each speed comparison runs both wheels in one session, order balanced (base, fix, fix, base), one process per arm, no other GPU work during the arm (the lab's GPU lock was held). Arms last about four minutes each. The host's thermal state was not controlled or recorded. The spread between the two arms of one kind is about 1% of decode time.

## Device reopen

Every arm opens the Vulkan device in a fresh process, and `vulkaninfo` listed the device normally after the last run (exit code 0). A failed arm (the out-of-memory ones) did not prevent the next process from opening the device.

## Limits

One model, one prompt family. The in-tree C++ test case for `round_size` was not built or run by the author; a standalone program with the same function body passes its assertions over 1027 sizes (worst waste 12.4% against 99.9% for the old rule). Residency 0.5 still exceeds a 16 GB host's heap even with exact sizing (48 MiB arrays, 8.1 GB of slot arrays) and is not addressed here.
