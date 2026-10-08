# llama.cpp on the Honeykrisp stack: M1 Max before/after

Claim backed: on an Apple M1 Max (G13C, 64 GB) running Linux, the same
llama.cpp Vulkan binary runs Qwen3-30B-A3B UD-IQ1_S (128 experts, 8
active) far faster on the Honeykrisp driver stack under review than on
the stock Mesa driver. Only the Vulkan driver changes between the two
rows.

| driver | prefill (pp512) | decode (tg64) |
|---|---|---|
| stock Mesa 26.2.2 (system driver) | 55.4 tok/s | 1.53 tok/s |
| Honeykrisp build git-6543eeb7df, this stack | 236.5 tok/s | 27.4 tok/s |

Decode is 17.9x faster. Prompt processing is 4.3x faster.

## Setup

| item | value |
|---|---|
| llama.cpp | commit `65840ed`, Vulkan build, includes `llama-bench` |
| model | Qwen3-30B-A3B UD-IQ1_S GGUF |
| driver A | Mesa 26.2.2 asahi, the system driver (`VK_DRIVER_FILES` unset) |
| driver B | Honeykrisp from [joshuaswarren/mesa-1](https://github.com/joshuaswarren/mesa-1), branch `honeykrisp-omarchy-v3` (`HK_LARGE_CONSTANTS` default on), selected with `VK_DRIVER_FILES` |
| method | `llama-bench -p 512 -n 64` on one host, one binary, one driver per session; repeated runs agreed within noise |

## Same driver, one flag

The second driver with `HK_LARGE_CONSTANTS=0` (the old lowering, no
rebuild) isolates the largest single change:

| driver B, flag | prefill (pp512) | decode (tg64) |
|---|---|---|
| `HK_LARGE_CONSTANTS=0` | 185.5 tok/s | 1.5 tok/s |
| `HK_LARGE_CONSTANTS=1` (default) | 237.0 tok/s | 26.6 tok/s |

The IQ quant grid tables move from per-thread scratch to constant
memory. In the MUL_MAT_ID iq2_xs microbenchmark the kernel goes from
2,290 to 102 microseconds per call, and the IQ mat-vec kernels run 5 to
24 times faster.

## Second data point: Llama-3.1-8B IQ2_M

Measured in the [omacom/mesa#6](https://github.com/omacom/mesa/pull/6)
review thread, same chip:

| check | old lowering | this stack |
|---|---|---|
| decode tg32 | 0.94-0.95 tok/s | 12.78 tok/s (12.74-12.89 over 3 runs) |
| prompt pp512 | 123.5-124.2 tok/s | 149.6 tok/s |

## Correctness

- llama.cpp `test-backend-ops`: MUL_MAT 1,684 of 1,684, MUL_MAT_ID 939
  of 939.
- Greedy token digests are unchanged on M1, M1 Max, and M2 Max.
- `tests/large_constants.py` in the driver tree: 0 mismatches of
  131,072 samples on M1 (G13G), M1 Max (G13C), and M2 Max (G14C).

## Reproduce

```bash
git clone -b honeykrisp-omarchy-v3 https://github.com/joshuaswarren/mesa-1
cd mesa-1
meson setup build -Dvulkan-drivers=asahi -Dgallium-drivers= -Dplatforms= -Dvulkan-layers=
ninja -C build
# point VK_DRIVER_FILES at the Honeykrisp ICD json the build emits
llama-bench -m Qwen3-30B-A3B-UD-IQ1_S.gguf -p 512 -n 64
HK_LARGE_CONSTANTS=0 llama-bench -m Qwen3-30B-A3B-UD-IQ1_S.gguf -p 512 -n 64
```

The review stack is omacom/mesa [PR #3](https://github.com/omacom/mesa/pull/3),
[#4](https://github.com/omacom/mesa/pull/4),
[#5](https://github.com/omacom/mesa/pull/5), and
[#6](https://github.com/omacom/mesa/pull/6), all open.
