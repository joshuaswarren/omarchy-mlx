# Register-blocked f32 matmul when no cooperative-matrix route takes the product

Date: 2026-11-13. This receipt backs `MatmulRbF32`: the f32 build of `shaders/matmul_rb.comp`, which `dispatch_matmul` now takes for a float32 matmul with `matrix_m >= 32` and no bias when neither `MatmulDirectF32` nor `MatmulF32Coopmat` takes it.

## Why

Both f32 cooperative-matrix routes need `cooperative_matrix_f32_8`. A driver without `VK_KHR_cooperative_matrix` reports it false, so every f32 product there ran the 16x16 tile (`MatmulF32`, one output per thread). The Borealis driver on the M1 Pro, M2, M4 Pro and A18 Pro does not list the extension, and the collector's fp32 benchmark read 0.38 TFLOP/s at 1024 on the M1 Pro and M4 Pro, 0.21 on the M2 and 0.13 on the A18 Pro. f16 already had the 64x64 register-blocked tile for this case (`MatmulRbF16`).

## Exactness

The f32 tile keeps `MatmulF32`'s arithmetic: one float accumulator per output, k in ascending order, zero padding to a multiple of 16, then `alpha * acc`. The stored bits are the same.

- `omarchy_matmul_family_tests`, new case "register-blocked f32 matmul matches the 16x16 tile bit for bit": five shapes (k tail 1052, transposed b with an n tail, column-major lhs, 256 square, both transposed with odd sizes), batch of three. It compares the 64x64 tile at full m against the 16x16 tile on a 16-row slice, and checks every row against a float64 host reference. It skips on a device with the f32 cooperative matrix, where f32 takes those routes instead. It passes on the M1 Pro and the M2, along with the existing f16 case (`logs/`).
- `tools/mm_bench.py` hashes every result. In all 8 shapes below, the sha256 under `MLX_OMARCHY_NO_MATMUL_RB_F32=1` (16x16 tile) equals the default (64x64 tile) on both machines. The M1 Pro and M2 hashes are also equal to each other.
- The f16 build of `matmul_rb.comp` compiles to the same SPIR-V as before the change (glslang, Vulkan 1.3 target).

## Machines and stack

| Board | Machine | OS | Vulkan device |
|---|---|---|---|
| G13S | MacBook Pro 16-inch 2021, M1 Pro (`apple,j316s`) | Fedora 45, Linux 7.3.0 | `Apple M1 Pro (G13S C0)`, Borealis, driver ID 35 |
| G14G | MacBook Air 13-inch 2022, M2 (`apple,j413`) | Fedora 44, Linux 7.3.0 | `Apple M2 (G14G B0)`, Borealis, driver ID 35 |

Neither device lists `VK_KHR_cooperative_matrix`. The wheel and test binary were built once from this branch (`04001f1a`) on Fedora 44 aarch64 with `-DCMAKE_INSTALL_LIBDIR=lib` and installed into a copy of each machine's v0.7.41 venv (Python 3.14). Both arms of the A/B use the same wheel. `MLX_OMARCHY_NO_MATMUL_RB_F32=1` selects the 16x16 tile, which is the dispatch `main` makes for these shapes. `MLX_OMARCHY_BATCH_WORK` was 20000 (M1 Pro) and 10000 (M2) in both arms. Desktop session with no other GPU work, `schedutil` governor.

## Speed

`tools/mm_bench.py` uses the collector's protocol: 2 warm-ups, then 8 timed `a @ b` with eval and synchronize, median. Random normal f32 operands. Order: tile, 64x64, tile, 64x64. Each cell is the mean of the two runs per arm (each run a median of 8), in TFLOP/s:

| Shape | M1 Pro tile | M1 Pro 64x64 | ratio | M2 tile | M2 64x64 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1024, a @ b | 0.382 | 0.864 | 2.26 | 0.210 | 0.544 | 2.59 |
| 2048, a @ b | 0.400 | 1.074 | 2.68 | 0.217 | 0.597 | 2.75 |
| 4096, a @ b | 0.403 | 1.088 | 2.70 | 0.218 | 0.597 | 2.73 |
| 1024, a @ b.T | 0.375 | 0.883 | 2.35 | 0.207 | 0.548 | 2.64 |
| 1024, a.T @ b | 0.370 | 0.883 | 2.39 | 0.207 | 0.547 | 2.64 |
| 2048, a @ b.T | 0.395 | 1.083 | 2.74 | 0.214 | 0.595 | 2.77 |

At 256 and 512 one call is under 1.5 ms and is dominated by submission time, and the two runs of one arm differ by up to 3x. Those rows are in the logs and are not used here.

The change touches only f32 matmuls with at least 32 rows and no bias. The 4-bit and bf16 decode and prefill paths of the LLMs in `receipts/` do not dispatch them, so no tokens-per-second change is claimed.

## Files

`ab/` (M2 JSON, one file per run), `logs/` (test and timing output from both machines; the M1 Pro JSON stayed on the machine, and its log carries the same rows), `tools/mm_bench.py`, `SHA256SUMS`.
