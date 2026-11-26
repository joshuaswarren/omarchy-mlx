# gather_qmm word-per-lane 2-bit route: kernel gate, layer A/B, hidden-state fidelity, device doctest

This receipt records the hardware runs behind the numbers in `docs/KNOWN-ISSUES.md` for the opt-in switch `MLX_OMARCHY_GATHER_QMM_WORD2=1`. The route is off by default. `SHA256SUMS` lists the files here.

The files are copies of the run outputs with these edits. Paths are shortened to `<scratch>` and `~`. Boot ids are replaced by `<id>`. Lane codes in labels are replaced by `lane`. Host names are replaced by `host`. Line 2 of `fidelity/ticket.out` and of `device-doctest/run.ticket.out`, the guard's echo of the kernel command line, is removed.

## Source

The shader `overlay/mlx/backend/omarchy/shaders/gather_qmm_word.comp` has git blob `6e67fb2ff5ed5f3f8069f4382420b37aa2832dfe` at every commit below. All runs exercise the same shader.

| Run | Source commit | Library |
|---|---|---|
| Kernel gate and per-call cost | `1860679` (`mlx 0.32.4.dev202610162350+1860679`) | `libmlx.so` prefix `a78282e363bfbb7a` |
| Layer A/B | `1860679` | `a78282e363bfbb7a` |
| Hidden-state fidelity | `37ff5f09`, wheel `0.32.4.dev202610171031+37ff5f0` | sha256 `dfbef1940389d76202a04f44b3c00379c73ca9cc6ad44a7d19e6974b4dbb471e` |
| Device doctest | `127a6110` | test binary sha256 `85ee7605de04aeb033fbf91d371f63451ab0a4a93fc692ef5c548758358b882a` |

Kernel ids are labels. They move when kernels are appended. The route is named by its enum name `GatherQmmWord2BF16`. Its id was 619 at `1860679`, 635 at `37ff5f09` and 636 at `127a6110`.

## Machine and driver

- 14-inch M2 Max (G14C), Linux kernel 7.1.12-2-20261013.18-sep-ARCH, CPU governor performance.
- Gate, layer A/B and fidelity: the installed Vulkan driver package (Mesa git 69bde15340), version `1:26.2.4.git69bde15340-2`, library sha256 prefix `40282811c6e14ef9`.
- Device doctest: library sha256 prefix `c499962c742f399a`, Mesa git `a5b1797446`.
- Platform firmware of the 14-inch M2 Max, read from its device tree: OS firmware version 13.5, system firmware version 27.0, iBoot-8422.141.2, boot loader v1.6.1-omarchy.aurora14.

## Device identity, reopen and thermal procedure

- The device is named by mlx as `Device(gpu, 0)` in the gate and the layer A/B outputs. The Vulkan device name string was not recorded by these tickets.
- No Vulkan device reopen was run in these tickets. The GPU timeout counter was read before and after the gate, fidelity and device doctest tickets and stayed at 0.
- No temperature or fan readings were taken. The thermal procedure is the guard's load settle before each ticket and the performance CPU governor. The load averages are in the sections below.

## Model

Muse Spark Open 2-bit, layers 14 to 17 of 62. The expert tensors have 256 experts, N 4096, K 4096 and group 128. Each token uses 8 experts. Snapshot `model.safetensors.index.json` sha256 `bc53f5a5f9afb6960c828f2192eef64898a1de173a3a34e6c46b19625f8629eb`. Tensor list sha256 `c5b35cff78450ff3b29ee1ed073b4a9644ade4d7007b40045c2af41db6af9eda`.

## 1. Kernel gate and per-call cost (`gate/gword2.txt`)

The gate has six shapes and 36 cases. Each case is bf16 with 8 experts per call. The x layouts are broadcast and per-expert. The index patterns are sorted, random and repeated.

In every case, the word kernel and the 8-output kernel have 0 outputs beyond 1.5 bf16 steps of a float64 contraction. The largest error is 0.50 steps for both.

The dispatch trace names the kernel of each route in every case. With the switch on, it is the word kernel (id 619 in that build). With the switch off, it is the 8-output kernel (id 612) in 30 cases. It is the one-output kernel (id 581) in the 6 cases of the shape with N 100, because N is not a multiple of 8.

Per-call cost at the Muse expert shape (8 experts of 4096 by 4096, group 128, 37748736 bytes per call) is the median of 4 blocks. The word kernel takes 290.9 us (129.8 GB/s). The 8-output kernel takes 1184.6 us (31.9 GB/s). The ratio is 4.07. The load average was 1.76 at the start of the ticket. The file lists the block medians, minima, maxima and the smaller shapes.

Command: `python3 gword2_gate.py`, in one guard ticket. The GPU timeout counter was 0 before and after.

## 2. Layer A/B (`layer-ab/`)

The run covers layers 14 to 17 of the real snapshot with one decode row. Each arm runs 12 steps per round, for 3 rounds that alternate the two arms. The harness drops the first 2 steps of a round, so a step median is the median of the other 10. Compile is disabled and `--no-ops` is set. Arm "sub-on" runs the word route. Arm "word2-off" runs the 8-output kernel. The A/B ran twice in the same boot.

| Run | Word route, step median per round (ms) | Per layer | 8-output kernel, step median per round (ms) | Per layer |
|---|---|---|---|---|
| 1 (`run1.*`) | 26.705, 26.155, 26.56 | 6.64 ms, spread 0.55 ms | 37.85, 37.91, 37.975 | 9.477 ms, spread 0.13 ms |
| 2 (`run2.*`) | 26.705, 26.58, 26.375 | 6.645 ms, spread 0.33 ms | 37.885, 37.825, 38.04 | 9.471 ms, spread 0.21 ms |

The numbers in `docs/KNOWN-ISSUES.md` (6.64 and 9.48 ms per layer) are run 1. The last hidden state differs by at most 0.03125 in both runs. In run 2 the hidden state has a maximum of 7.469 and an rms of 0.656.

Command: `muse-layer-bench2.py <snapshot> 14 17 --arms sub-on,word2-off --rounds 3 --steps 12 --no-ops`, through the pinned wrapper `benchpy.sh`. The bench script has sha256 prefix `d2965d6a6c9f0d99` in run 1 and `641ddd40eabd8c13` in run 2.

## 3. Hidden-state fidelity (`fidelity/`)

The run was preregistered and had one look. It covers 32 windows (128 to 159). Each window is a 2-token prefill and one decode row through layers 14 to 17. Five arms run in separate processes. S has the switch unset. R has the switch on. F is a float64 contraction of the same stored words. S2 and R2 repeat S and R. Routing is not pinned.

- S equals S2 and R equals R2 bit for bit in 32 of 32 windows. The inputs sha256 `18acd6d6cf33c74b72e5d0e8d1bb8abacbbab6d5c58c985e9ede523ab0e77571` is equal in all arms. F equals neither S nor R in any window.
- The dispatch trace of window 0 shows the routes. S has 12 dispatches of kernel 612 and 0 of `GatherQmmWord2BF16` (635 in that build). R has 12 of 635 and 0 of 612. F has neither. The totals are 612, 612 and 568 dispatches.
- The registered verdict is WIDER, so there is no ACCEPT. The mean u is +0.0555 with the interval [-0.2331, +0.3441]. The rule needs the interval below ln 1.05.
- Three of 32 windows had a top-8 routing flip. Windows 142 and 145 flipped in R and window 157 in S. The 29 windows with shared routing give ACCEPT-NOT-WORSE, with mean u -0.0241 and the interval [-0.0810, +0.0328].
- Per call, the worst single-call rms error over the rms reference is 1.329e-4 in S and 1.328e-4 in R, over 384 calls per arm (12 calls in each of 32 windows). The mean rms error over all calls is 5.027e-6 in S and 4.455e-6 in R. At most 0.0244 percent of a call's 32768 outputs differ from the float64 reference. `fidelity/per-call-worst.txt` holds these numbers and the sha256 of the S and R npz files. `fidelity/per_call_worst.py` computes them from the `diag` arrays of those two files.

Command: `word2-run2.sh <out> <tree> dfbef1940389d762 635 612 128 160`. The wrapper has sha256 `d42e33bf1936c55337ebd4e53de5d860dda847ced2300fb5305718bca4376e73`. The harness has sha256 `920367b59ef425b7fdeccb26638d20abd70081c8d5c3a3db6732b9a45df57126`. The compare script has sha256 `b0c5a3c4fe3210c56bda2c736e95c401f1ef1a8c06fd9d0205f05f73adf83413`. The ticket settled at load 0.47. The GPU timeout counter stayed at 0.

## 4. Device doctest (`device-doctest/`)

`omarchy_matmul_family_tests -tc='gather_qmm bf16*'` ran 2 of 2 test cases and 1095 of 1095 assertions, with 0 failed. `-tc='gather_qmm bf16 2-bit*'` alone ran 1 of 1 case and 666 of 666 assertions. The 666 includes the four checks that the word kernel was dispatched. Those checks run only on a 32-lane subgroup device. The ticket settled at load 0.40. The GPU timeout counter stayed at 0.
