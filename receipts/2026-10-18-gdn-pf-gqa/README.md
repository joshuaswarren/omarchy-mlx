# GatedDeltaUpdate decode: opt-in native GQA for the untiled prefetch kernel

This receipt records the hardware runs behind `MLX_OMARCHY_GDN_PF_GQA=1`. The switch is off by default. `SHA256SUMS` lists the files here, and `SHA256SUMS-arrays-not-copied.txt` in each run directory lists the hashes of the large float arrays that are not copied.

The files are copies of the run outputs. Boot ids are replaced by `<id>`. The scripts in `tools/` are the scripts that ran, with the author's working-directory names changed to `gdn`. Their original sha256 prefixes are in the table under "Tools".

## What the switch does

On the f32 `A_log` raw-gates route (the Qwen3.5-9B route) with 16 key heads and 32 value heads, the backend expands q and k to the value-head count with two copies per linear layer per token. The tiled kernel reads the key head itself and skips the copies. The untiled prefetch kernel (`MLX_OMARCHY_GDN_DECODE_TILE=0`, or a legacy G13 part by default) does not. The new kernel `GatedDeltaDecodeBF16PfComposedGqa` is the untiled prefetch kernel in composed order with the key-head index of the tiled one, built from `gated_delta_decode_perrow_pf.comp` with `-DGDN_NATIVE_GQA_INDEX`. It applies only where the tile is off, for f32 `A_log`, `Hk != Hv` and an h0 offset that is a multiple of 4. Nothing else changes.

On main 835b12fe its ComputeKernel id is 637 (`GatherQmmWord2BF16` is 636). The run `m2-rebased-wheel-9fde0d95` shows 637 in the dispatch trace. The three earlier runs used a wheel built before the rebase, where the same kernel had id 636. The arithmetic of the kernel did not change in the rebase.

Arms in every run: A is the default (the tiled kernel on the M2 and the 16-inch, the untiled kernel on the 13-inch). B is `MLX_OMARCHY_GDN_DECODE_TILE=0`, the untiled prefetch kernel with the two expansion copies. C is `MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_GQA=1`.

## Source

- Commit 6c863089 (main 878816d6 plus the change): wheel `mlx_omarchy-0.32.4.dev202610180652+6c863089`, sha256 `37533af3d7474a6e6eb9ecb2a5172123075821dadd8636d0ff65f355e2e566b3`, libmlx sha256 prefix `88c62f2ed14fe7b2`. The same wheel ran on all three machines.
- Commit 9fde0d95 (the same change rebased on main 835b12fe): wheel `mlx_omarchy-0.32.4.dev202610180805+9fde0d95`, sha256 `db3228fb41ad5f56253820d911d106716732a2a89f14c1f335620a5c73963c4f`, libmlx sha256 `1334699b94b07a5d626bec2119cdf71e7b6b4b12137ebbc9ed844b4620b0d09a`. Both wheels were built on the 14-inch M2 Max.
- Harness: `scripts/bench_decode.py` sha256 prefix `8220064c5ea3aaaf`, `scripts/mlx_provenance.py` `21337ddd29e79291`, patches tree `a003c55941c6c106` (76 files), mlx-lm 0.31.3 with those patches. Every bench and recorder run printed `verified=match` from the provenance check (the loaded library equals the wheel record).

## Machines

| Machine | Chip id in the output | Kernel | Vulkan driver | Model repo and snapshot |
|---|---|---|---|---|
| 14-inch M2 Max | Apple M2 Max (G14C B1) | 7.1.12-2-20261013.18-sep-ARCH | Mesa 26.2.4, driver package git a5b1797446, library sha256 prefix `c499962c742f399a` (read after the runs, same boot) | mlx-community/Qwen3.5-9B-4bit `8b2b98c00a6b4d291155e4890773ca8f769aee53` |
| 16-inch M1 Max | Apple M1 Max (G13C C0) | 7.1.12-2-12.6-sep-ARCH | Mesa 26.2.4, driver package git 69bde15340, library sha256 prefix `40282811c6e14ef9` | mlx-community/Qwen3.5-9B-MLX-4bit `938d8919941c6e7efd3c7150eff7fe9d12afa631` |
| 13-inch M1 | Apple M1 (G13G B1) | 7.1.12-2-20261013.18-sep-ARCH | Mesa 26.2.4, driver package git 69bde15340, library sha256 prefix `40282811c6e14ef9` | mlx-community/Qwen3.5-9B-4bit `8b2b98c00a6b4d291155e4890773ca8f769aee53` |

All three ran with the performance CPU governor and no `GDN_`, `MLX_`, `HK_` or `AGX_` variable in the environment (the run log counts them: 0). The snapshots are the Hugging Face revision ids of the 4-bit models.

Device and firmware identity. The chip id in the table is the Vulkan device name string: every bench output carries it in its `device` field, which the harness takes from `mx.metal.device_info()["device_name"]`, and the backend fills that field with the `deviceName` of the Vulkan physical device. The driver identity is the Mesa version, the driver package and the library hash in the table. The firmware values were read from the device tree of each machine after the runs, on the same boots: the 14-inch M2 Max has OS firmware 13.5, system firmware 27.0 and iBoot-8422.141.2; the 16-inch M1 Max has OS firmware 13.5, system firmware 26.6.2 and iBoot-8422.141.2; the 13-inch M1 has OS firmware 13.5, system firmware 27.0 and iBoot-8422.141.2.

## Commands

Each run is one driver script from `tools/`: `gdn-pf.sh` (14-inch M2 Max), `gdn-pf-m1.sh MODEL TAG` (the two M1 machines) and `gdn-pf2.sh` (the rebased wheel). The drivers print their own start line, wheel hash and provenance into `run.log`. Per process:

- Speed: `bench_decode.py --model MODEL --prompt PROMPT --tokens 64 --wheel WHEEL`, prompt P1 `Explain in plain words how a bicycle works.` and P2 `Write a short story about a lighthouse keeper.`, 5 rounds per prompt and arm, the order rotating A B C, B C A, C A B, A B C, B C A. The `decode` rate is the 63 decoded tokens after the first. The rebased run has one round of P1 and is not a speed result.
- Numerics: `gdn_logits.py --model MODEL --prompt P1 --tokens 128 --full-steps 32` for arms A, B, C and C again (C2); the M2 run also ran P2. It writes the generated ids, the top-20 values and the full log-probability vectors of the first 32 steps.
- Census: the same recorder with `MLX_OMARCHY_TRACE_DISPATCH=1` at 6 and 10 tokens (the M2 run also at 128). Dispatches per token are the difference of the 10-token and 6-token runs divided by 4.

## Results

Decode tok/s, median of 5 rounds (range in the run directories). Every C sample is above every B sample, and every B sample above every A sample, where the table says "yes".

| Machine | Prompt | A | B | C | B vs A | C vs B | C vs A | every C above every B / every B above every A |
|---|---|---|---|---|---|---|---|---|
| 14-inch M2 Max | P1 | 40.565 | 43.287 | 43.871 | +6.71 % | +1.35 % | +8.15 % | yes / yes |
| 14-inch M2 Max | P2 | 40.503 | 43.266 | 43.919 | +6.82 % | +1.51 % | +8.43 % | yes / yes |
| 16-inch M1 Max | P1 | 39.285 | 42.290 | 42.749 | +7.65 % | +1.09 % | +8.82 % | yes / yes |
| 16-inch M1 Max | P2 | 39.312 | 42.211 | 42.742 | +7.37 % | +1.26 % | +8.73 % | yes / yes |
| 13-inch M1 | P1 | 12.045 | 12.061 | 12.073 | +0.13 % | +0.10 % | +0.23 % | no / no |
| 13-inch M1 | P2 | 12.052 | 12.063 | 12.094 | +0.09 % | +0.26 % | +0.35 % | no / no |

On the 13-inch machine arms A and B are the same route (the legacy G13 part already runs the untiled kernel with the copies), so B vs A is the spread of two identical routes: up to 0.13 %. The C gain there is at the edge of that spread and is not claimed. The other two machines default to the tiled kernel; C is 8 to 9 % faster than that default and 1.1 to 1.5 % faster than the untiled route that pays the copies.

64-token output digests are equal in all arms and on all three machines: `0a1a64828ea650e8` (P1) and `fd33be8b314efb95` (P2).

Dispatches per token (kernel ids of the wheel `6c863089`; the new kernel is 636 there and 637 in the rebased run):

| Machine | Arm | Dispatches per token | Decode kernel x layers | CopyGeneral (id 58) per token |
|---|---|---|---|---|
| 14-inch M2 Max and 16-inch M1 Max | A | 306 | 514 x 24 (tiled composed) | 8 |
| 14-inch M2 Max and 16-inch M1 Max | B | 354 | 516 x 24 (untiled prefetch composed) | 56 |
| 14-inch M2 Max and 16-inch M1 Max | C | 306 | 636 x 24 (new kernel) | 8 |
| 13-inch M1 | A and B | 370 | 516 x 24 | 56 |
| 13-inch M1 | C | 322 | 636 x 24 | 8 |
| 14-inch M2 Max, rebased wheel | A / B / C | 306 / 354 / 306 | 514 / 516 / 637 x 24 | 8 / 56 / 8 |

The 128-token traced runs on the M2 total 40743 dispatches for arms A and C and 46935 for arm B.

Numerics: prompt P1, 128 greedy tokens, the full log-probability vector (248320 entries) of the first 32 steps, for the pairs A-B, B-C, C-C2, A-C and A-C2 on each machine and on the rebased M2 run: generated ids equal 128 of 128, top-1 and top-5 equal 128 of 128, largest absolute difference 0.0 over the full vectors and the top 20, 0 differing elements, KL 0.0. The M2 run repeats this for P2 with the same result. The `full.npy` hashes in `SHA256SUMS-arrays-not-copied.txt` are equal across arms for each prompt on the M2 (P1 `0e39a2478649b53b023e...`, P2 `2fed2a7ba0c57591bd88...`).

`recompute.txt` in each run directory is the output of the scripts `pfre.py` and `pfre2.py` (14-inch M2 Max), `pfre-m1.py` (the two M1 machines) and `pfre-pf2.py` (rebased run) in `tools/`. They read the raw files of the run directory: the bench `t-*.out` files, the census `hist-*.txt` files and the numerics directories. Their path names point at the author's working directory and are shown as `<run directory>`; the `full.npy` arrays are not copied, so the numerics comparison needs the arrays from the original run (their hashes are listed).

## Procedure

- Timing: the guard's idle settle before each ticket, the performance governor, the arm order rotated every round, one process per sample. The load average at the start of each run: 0.31 (14-inch M2 Max), 0.48 (16-inch), 0.36 (13-inch), 0.49 (rebased M2 run). No temperature, fan or throttle reading was taken.
- Reopen and recovery: no dedicated reopen test (close and reopen the device inside one process) was run. Each run is a sequence of separate processes, and every process opened the Vulkan device from the start after the one before it had exited: 47 processes on the 14-inch M2 Max, 40 on the 16-inch M1 Max, 40 on the 13-inch M1 and 13 in the rebased M2 run. All of them finished with exit code 0 and printed `verified=match`; the run logs count 47, 40, 40 and 13 of these lines and no non-zero exit code. The guard counted the GPU timeout lines of the kernel log (`Job timed out on the DRM scheduler` and `GPU timeout`) before and after each ticket: 0 and 0 on the 14-inch M2 Max (tickets ending 07:07:41Z and 08:17:00Z), 0 and 0 on the 16-inch M1 Max (ticket ending 07:43:09Z) and 0 and 0 on the 13-inch M1 (ticket ending 07:46:02Z). The kernel log of each machine, read after the runs, still shows 0 such lines.
- Dispatch trace: the census above, from `MLX_OMARCHY_TRACE_DISPATCH=1`.

## Limits

- One boot per machine, two prompts of 64 decoded tokens for speed and one prompt of 128 tokens (two on the M2) for numerics. Model numerics beyond that are open.
- The route is the f32 `A_log` route. The 27B model has bf16 `A_log` and does not take it.
- The 16-inch and the other two machines use different model repositories (see the table). Whether their weights are the same files was not compared; the digests of the output ids agree.
- Speed numbers belong to the wheel `6c863089`. The rebased wheel ran one speed round of P1 (A 40.41, B 43.17, C 43.83 tok/s) to check the digest and is not a speed result.

## Tools

| File | sha256 prefix of the original script that ran |
|---|---|
| `tools/gdn-pf.sh` | `c9b038c287849f27` |
| `tools/gdn-pf-m1.sh` | `094b10a99b290cae` |
| `tools/gdn-pf2.sh` | `f7e02c35014c81de` |
| `tools/gdn_logits.py` | `4ac2745a45594687` |
