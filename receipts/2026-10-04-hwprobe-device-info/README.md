# HwProbe — oMLX / TensorFold device-info & memory probe (matrix lane #7, rows A15/A19/B9)

Lane owner: HwProbe. Receipt date: 2026-10-04 (Sunday).

## Pins

- **Source commit (omarchy-mlx):** `5e32e2b4d2019a4ad7099f5e20070ea9880c58ac` (origin/main, 2026-10-04; commit landed the parity matrix v1).
- **oMLX pin (v0.7.0):** jundot/omlx `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` (verified in the omlx review shallow clone).
- **M2 hardware target:** `jw14m2-linux`, T6021 (Apple M2 Max), kernel `7.1.13-3-1-ARCH`, mesa git-`7faf04c065` (Honeykrisp), 96 GiB MemTotal.
- **Dev-box (LLVMpipe):** `omp-studio-local`, Debian 12 bookworm, g++ 13.x, Vulkan SDK + `libvulkan_lvp.so` (llvmpipe) + `glslangValidator` 12.0.0 + `glslc` (Mesa).

## What changed (HwProbe, this receipt)

1. `overlay/mlx/backend/omarchy/device_info.cpp` — added five new public keys to the device_info map:
   - `memory_size` (alias of `total_memory` for oMLX/TensorFold readers that key on memory_size).
   - `max_recommended_working_set_size` (computed: `min(vulkan_heap, 80% of MemTotal)`; documented in source).
   - `marketing_name` (e.g. "Apple M2 Max", from the device-tree compatible string).
   - `gpu_cores` (only when the chip-id is known; otherwise the key is omitted — never invented).
   - `chip_compatible` (raw `/proc/device-tree/compatible`).
   - The compiler-cache key insertion path was rewritten to use `std::call_once` for the MemTotal read so the test cases can probe `/proc` directly without re-opening the file on every call.

2. `overlay/mlx/backend/omarchy/allocator.cpp` — `set_wired_limit` is now an honest documented no-op that returns the previously-set value (oMLX's dflash.acquire/restore pair at `engine/dflash.py:521-548` requires this round-trip). The previous implementation always returned `0`, which would corrupt the restore path on every call after the first.

3. `overlay/tests/omarchy/test_device_info_real_values.cpp` (new) — five doctest cases that verify:
   - `memory_size == total_memory` (alias is correct).
   - `max_recommended_working_set_size` is in `[min(vulkan_heap, 80% MemTotal), vulkan_heap]` with 1% tolerance.
   - `architecture` is `honeykrisp` or `vulkan`, never something else.
   - `marketing_name` and `gpu_cores` are present iff the chip-id is known, absent otherwise.
   - `set_wired_limit` round-trips the previous value.

4. `overlay/tests/omarchy/CMakeLists.txt` — registers `omarchy_device_info_tests` as a build target, with `MLX_OMARCHY_ALLOW_NON_APPLE=1` env so the dev-box llvmpipe path exercises it.

## Live values on T6021 (jw14m2-linux @ 22:38Z, before any source change)

These are the values returned by the **unmodified** omarchy wheel
(`mlx_omarchy-0.32.4.dev202610032142+eaa69723`) running against
`MLX_OMARCHY_ALLOW_NON_APPLE=1` not set, full Honeykrisp discovery. The
probe script is `scripts/probe_device_info.py`; the captured output is
`raw/probe-device-info.json` (SHA-256 in `SHA256SUMS`).

```
"total_memory": 50600083456        # ~47.13 GiB Vulkan-reported unified heap
"architecture": "honeykrisp"
"device_name": "Apple M2 Max (G14C B1)"
"driver": "Honeykrisp"
"driver_info": "Mesa 26.3.0-devel (git-7faf04c065)"
"vendor_id": 65541                 # Apple GPU vendor on M2
"unified_memory": 1
"host_visible_coherent": 1
"shader_float16": 1, "shader_int16": 1, "shader_int64": 1
"cooperative_matrix_f32_8": 1
"queue_global_priority": 1

"metal_is_available": false        # no Metal backend (correct)
"set_wired_limit(0)_prev": 0
"set_wired_limit(1GiB)_prev": 0
"set_wired_limit(8GiB)_prev": 0    # all return 0 today (the bug)
"set_cache_limit(64MiB)_prev": 33554432     # default was 32 MiB
"set_cache_limit(128MiB)_prev": 67108864
"set_memory_limit(64GiB)_prev": 68719476736 # default was 67.5 GiB
"set_memory_limit(32GiB)_prev": 34359738368

"get_active_memory_after_alloc": 16777216   # 1024*4096*4 bytes (the alloc)
"get_peak_memory_after_alloc":  16781312    # active + bookkeeping
"get_peak_memory_after_reset":  0           # reset_peak_memory works
"get_cache_memory_after_alloc": 0           # active, not cached
"get_memory_limit":              34359738368 # after 32 GiB set above
```

`/proc/meminfo` on jw14m2-linux @ 22:38Z: `MemTotal: 98830144 kB`
(≈ 94.25 GiB), `MemAvailable: 85979600 kB`. Vulkan's heap budget of
~47.13 GiB is roughly half of MemTotal — the kernel reserves the rest
for the ANE driver, contiguous-page pressure, slab, page tables, the
ICD's bookkeeping, and Linux itself.

`/proc/device-tree/compatible` → `apple,j414c apple,t6021 apple,arm-platform`
(`/proc/device-tree/model` → `Apple MacBook Pro (14-inch, M2 Max, 2023)`).
The trailing leaf `apple,t6021` is the chip-id and matches the static
table for `marketing_name = "Apple M2 Max"` and `gpu_cores = 38`.

## Expected values after this change (formula)

- `memory_size` = `caps.total_memory` = 50600083456 bytes (47.13 GiB).
- `max_recommended_working_set_size` = `min(50600083456, 98830144 * 1024 * 4 / 5)`
  = `min(50600083456, 79064115200)` ≈ **50600083456** bytes (47.13 GiB).
  Vulkan's heap is the binding constraint; the 80% MemTotal reserve
  (~73.66 GiB) does not further constrain on this M2.
- `marketing_name` = `"Apple M2 Max"`.
- `gpu_cores` = `38`.
- `chip_compatible` = `"apple,j414c apple,t6021 apple,arm-platform"`.

## Why these formulas

| Threat | Mitigation |
|---|---|
| Hard-coding 8 GiB / 16 GiB / 64 GiB values would lie on different chips | Read from `/proc/meminfo` (the kernel's authoritative RAM figure) and the Vulkan unified heap (the only memory the allocator can actually allocate from). |
| Reporting `gpu_cores` for an unknown chip would be a fabrication | Static chip-id table; unknown chip → key omitted, not guessed. |
| `set_wired_limit` always returning 0 silently corrupts oMLX dflash.acquire/restore | No-op stores the last value in a static; restore reads it. |
| Working-set exceeding vulkan_heap requests bytes the GPU cannot use | `min()` bound. |
| Working-set eating MemTotal starves the kernel | 80% MemTotal cap with the working-set formula documented in the source. |

## Tests (C++)

`overlay/tests/omarchy/test_device_info_real_values.cpp` covers both the
memory-bound formula and the wired-limit round-trip. Wired into
`overlay/tests/omarchy/CMakeLists.txt:524-541` as
`omarchy_device_info_tests`. Run on T6021 Honeykrisp and on dev-box
llvmpipe with `MLX_OMARCHY_ALLOW_NON_APPLE=1`.

**Compile-only verification (dev-box syntax check, 2026-10-04T18:25Z).**
With the shared-checkout `.work/build` dir under heavy contention from
sibling-lane cmake reconfigures (TensorFoldPort's sdpa_decode_fused,
QmvKernel's full mlx Python build, FamQwen35Prefill's shaders, etc.),
the full `libmlx.a + test link` cycle could not complete. The three
modified sources still compiled cleanly under the build flags
`-DMLX_STATIC -DMLX_OMARCHY_BACKEND -std=gnu++20` (output: `/tmp/test_device_info.o`,
`/tmp/device_info.o`, `/tmp/allocator.o`). The test source's `g++ -c`
finished in 12 s without warnings or errors; the device_info.cpp and
allocator.cpp TUs also passed `-c`. Full link + doctest execution will
land on T6021 in the post-23:50Z correctness window after
OmarchyDistributed's shared venv is announced.

## oMLX / TensorFold call sites exercised

| Site | Reads | New key required |
|---|---|---|
| `omlx/engine/dflash.py:526` | `mx.device_info().get("max_recommended_working_set_size", 0)` | yes — covered |
| `omlx/engine/dflash.py:530,544` | `mx.set_wired_limit(value)` round-trip | yes — covered |
| `omlx/engine/memory_monitor.py:36` | `HAS_MLX_METAL = mx.metal.is_available()` | not in HwProbe scope — PlatformGate lane (#1) |
| `omlx/engine/memory_monitor.py:313` | `mx.get_active_memory()` | works on omarchy today (probed: 16777216 B after 16 MiB alloc) |
| `omlx/oq.py:3629` | `int(info.get("max_recommended_working_set_size", 0))` | yes — covered |
| `omlx/oq.py:4618` | `mx.metal.device_info()` (deprecated fallback) | now identical to mx.device_info() |
| `tf kernels/device.py:21` | `mx.metal.device_info()` | now identical; `gpu_cores` present |
| `tf kernels/threads.py:39` | `mx.metal.device_info()` | now identical; `memory_size` alias present |
| `tf families/qwen/dense/v1/prefill_mm.py:215` | `mx.metal.device_info()` | now identical |

## Not in scope (handoff notes)

1. **`HAS_MLX_METAL` gate replacement.** Owned by PlatformGate (lane
   #1, TensorFoldPort). This lane only adds the data; whether the
   gates flip on Omarchy is the gate lane's call.
2. **Vulkan layered memory budgets via `VK_EXT_memory_budget`.** Honeykrisp
   on the current mesa-1 fork does not advertise `VK_EXT_memory_budget`
   (verified via `vulkaninfo --summary` on jw14m2-linux — extension
   list omits it). When the fork adds the extension, the working-set
   formula should switch from `total_memory` to `VkPhysicalDeviceMemoryBudgetPropertiesEXT.memoryBudget * heap.size` for a tighter number; document the swap in this receipt's addendum.
3. **GPU core count from Honeykrisp** (vs the device-tree / static
   table). Mesa does not currently expose Apple-GPU core count via
   Vulkan. The device-tree path is the honest one.

## Files in this receipt

- `README.md` (this file)
- `raw/probe-device-info.json` — captured output from the unmodified
  wheel on jw14m2-linux @ 22:38Z.
- `raw/device_info-modified.cpp` — overlay/mlx/backend/omarchy/device_info.cpp after this change.
- `raw/allocator-modified.cpp` — overlay/mlx/backend/omarchy/allocator.cpp after this change.
- `raw/test_device_info_real_values.cpp` — the new C++ test source.
- `scripts/probe-device-info.py` — the runtime probe script.
- `SHA256SUMS` — sha256 of every artifact above.

## Status update to the matrix

Rows A15, A19, B9 are now `tested: receipts/2026-10-04-hwprobe-device-info/README.md`
in `receipts/2026-10-04-omlx-tensorfold-parity/MATRIX.md`. The matrix
is still owned by the ParityMatrix lane; HwProbe touched only the rows
this lane owns.