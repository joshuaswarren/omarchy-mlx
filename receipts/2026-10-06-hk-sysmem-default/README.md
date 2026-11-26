# 2026-10-06 HK_SYSMEM default — Honeykrisp heap raised for MLX processes

Lane: HeapSysmem (worker). Host for all runs: jw16 (`ssh jw16-2025`,
M1 Max T6001 G13C, kernel 7.1.12-2-11.38-sep-ARCH, MemTotal 65397232 kB
= 62.36 GiB). Change landed as commit `348fe2aac` (rebased content of the
probe commit `f53e1b6e6`; the five changed files are identical — `git diff
f53e1b6e6 348fe2aac -- overlay/mlx/backend/omarchy/device.cpp
overlay/mlx/backend/omarchy/device.h
overlay/tests/omarchy/test_device_info_real_values.cpp docs/install-omarchy.md
docs/serve.md` is empty). Private twin: notebook
`entries/HeapSysmem/20261006T2200Z-jw16-hk-sysmem-default.md` +
`artifacts/HeapSysmem/hk-sysmem-default/`.

## What the change is

`overlay/mlx/backend/omarchy/device.cpp`: before Vulkan loads, the backend
sets `HK_SYSMEM` for its own process when neither an explicit `HK_SYSMEM`
nor a drirc `heap_memory_percent` chose a heap:

```
heap = max(50% of MemTotal, MemTotal - 16 GiB)   rounded down to 1 MiB
heap = min(heap, 60 GiB)                          VA-window clamp
set    only when MemTotal > 32 GiB                (else keep mesa's default)
```

- `std::optional<uint64_t> default_hk_sysmem_bytes(uint64_t)` (device.h) is
  the pure formula; `mem_total_bytes()` reads /proc/meminfo;
  `drirc_names_heap_memory_percent()` scans the files mesa's xmlconfig.c
  reads (`$DRIRC_CONFIGDIR` dirs, else `drirc.d` + `/etc/drirc`, plus
  `$HOME/.drirc`, `$XDG_CONFIG_HOME/drirc`) for the option name.
- Only MLX processes see it: the env is set inside the backend; every other
  Vulkan application keeps the 50% heap.
- 16 GB-class hosts are byte-for-byte unchanged (nullopt keeps mesa's 50%
  default), as is the 16 GB M1: max(8, 15.1-16 <= 0) never beats 50%.

## Formula test (failing before / passing after)

Two doctest cases added to `overlay/tests/omarchy/test_device_info_real_values.cpp`
(`omarchy_device_info_tests`, registered in that directory's CMakeLists):

- cases over {15.1, 16, 32} GiB -> nullopt (stock heap kept; jwm1's measured
  15843696 kB MemTotal included); {48, 62, 64} GiB -> MemTotal - 16 GiB;
  {94, 96} GiB -> 60 GiB (clamp); 1 MiB rounding; live-MemTotal contract
  cross-check.
- RED: with pristine device.cpp/device.h the test does not compile —
  `error: 'default_hk_sysmem_bytes' is not a member of 'omarchy'`
  (jw16 /tmp/hk-red-build.log, copied to the artifact dir).
- GREEN (rebased tree `348fe2aac` content): omarchy_device_info_tests
  **8 cases / 31 assertions passed**.

An earlier green attempt failed to compile with `%` applied to the
`std::optional` return; fixed to `REQUIRE(has_value())` + `.value()` before
any result was recorded (dated in the notebook entry).

## Suites (jw16, battery flags, GPU work under gpu-turn)

Rebased tree, overlay-vs-.work full diff EMPTY before building.

| suite | result |
|---|---|
| omarchy_device_info_tests | 8/8 cases, 31/31 assertions, rc=0 |
| omarchy_runtime_tests | 49/49 cases, 22900/22900 assertions, rc=0 |
| omarchy_capability_sim_tests (6 profiles) | m1-honeykrisp-fork, m1-stock-no-coopmat, subgroup-size-64, small-shared-memory, no-cooperative-matrix, m1-g13-legacy — all rc=0 |

Pre-rebase runs (content-identical for the five files) were the same
8/8, 49/49 and 6/6 with the same rc=0; the rebased tree re-ran everything
above.

## Real-hardware proof (jw16, fresh processes, packaged ICD, HK_SYSMEM scrubbed)

Probe: `mx.zeros` 2 GiB chunks + `mx.eval`, arrays kept alive, MemAvailable
floor 12 GB, scripts `probe.py`/`probe-run.sh` in the artifact dir.

| arm | backend | env | result |
|---|---|---|---|
| before | b8 wheel (ade1656c3, today's main) | none | `FAILED: VK_ERROR_OUT_OF_DEVICE_MEMORY` at **30 GiB live** (16th chunk; 31.18 GiB stock heap) |
| after | new wheel f53e1b6 | none | backend set `HK_SYSMEM=49786896384`; reached **42 GiB live** (vkAllocateMemory OK), stopped by the **MemAvailable 12 GB floor**, not by the heap |
| explicit | new wheel | `HK_SYSMEM=4294967296` | kept untouched in `HK_env`; `FAILED ... OUT_OF_DEVICE_MEMORY` at 2 GiB live (heap stayed 4 GiB) |
| drirc | new wheel | `DRIRC_CONFIGDIR` with `heap_memory_percent` | `HK_env={}` in-process (default suppressed); failed at 30 GiB like `before` |
| Qwen3-32B-8bit | new wheel | none | **all 7 shards, 34.8 GB, LOADED** (`RESULT qwen3_32b_8bit weights_gb=34.8 LOADED`) — this model fails to load on the stock heap |

Direct heap number: the freshly built `mlx-omarchy-info` (HK_SYSMEM
scrubbed, packaged ICD) reports `total memory: 47480 MiB (device-local
heap)` = 49786896384 B, exactly `MemTotal - 16 GiB` (was 31920 MiB =
31.18 GiB).

Why "42" and not 44: another lane holds **22.85 GB of tmpfs** on jw16 right
now (`Shmem` in /proc/meminfo), so MemAvailable was 34-36 GB after all probe
processes exited; the owner-mandated 12 GB floor makes 44 GiB live
arithmetically unreachable (needs ~56 GB) regardless of the heap. The binding
limit is box occupancy, not the change: allocation SUCCEEDED 10 GiB past the
old failure point, and the heap itself is proven at 46.36 GiB above.

## The VA clamp — contradiction with the approved proposal text

The proposal said 96 GB -> ~78 GiB. The **per-process user VA window on
Apple Silicon under this mesa is ~64 GiB** (agx_device.c:632-649: the
pow2-rounded user window is halved twice for sparse-buffer emulation;
kernel vm_end fixed), measured live in HeapBudget probe B: 63 GiB committed
OK, the 64th allocation failed with "Failed to allocate BO VMA"
(`artifacts/HeapBudget/20261005-heap-budget/raw/runAB.log`: `ALLOC_FAILED
vk_result=-2 at_total_MiB=64512 count=63`). A 78 GiB heap could therefore
never be mapped; the implemented clamp is 60 GiB, so the 96 GB M2 gets
min(80, 60) = **60 GiB**, not ~78. jw16 (46.36 GiB) and 64 GB hosts
(48 GiB) are below the clamp; only 78+ GiB hosts are affected. Evidence:
`receipts/2026-10-05-heap-budget/`, lab note
`entries/mesa-bisect/20261006T1712Z-jw16-mesa-decode-bisect.md` (19:40Z
item: VA layout identical on T6001 and T6021, both uat_ias 39), artifacts
`mesa-bisect/decode-bisect/{heap_probe.c,heap-run.sh,heap-case.sh}`.

## Multi-process caveat (documented)

`docs/install-omarchy.md` gains the `HK_SYSMEM` paragraph (default, clamp,
precedence, MLX-only). `docs/serve.md` "Memory and context" gains the
caveat: the heap is a per-process ceiling, not a reservation — two MLX
processes side by side can still commit more than physical RAM together
(no swap; the OOM killer decides). oMLX multi-model serving needs a shared
budget; the serve's shared reservation transaction is what keeps managed
multi-model serving inside RAM.

## Compiled statement

| artifact | source | compiled | evidence |
|---|---|---|---|
| wheel `0.32.4.dev202610062222+f53e1b6` (sha256 `2e63d1c4…`) | f53e1b6e6 | **COMPILED** | DEV_RELEASE=1, pinned whole-encoder bundle (manifest `08769793…` verified against the runtime pin); venv-hs provenance `venv-hs mlx 0.32.4.dev202610062222+f53e1b6` |
| omarchy_device_info_tests / omarchy_runtime_tests / omarchy_capability_sim_tests | 348fe2aac content | **COMPILED, RUN — green** | jw16 `~/heap-work/logs/{build.log,device_info.log,runtime.log,capsim-*.log}`, overlay-vs-.work diff EMPTY |
| mlx-omarchy-info | 348fe2aac content | **COMPILED, RUN** | heap report 47480 MiB |
| probe arms + Qwen3 load | wheel f53e1b6 | **RUN** | `~/heap-work/logs/probe-*.log`, `heap-*.log` |

## Deviations

- The probe stop target 44 GiB was not reached (42 GiB; MemAvailable floor
  hit; jw16 tmpfs occupancy — numbers above). The floor itself is the
  owner's standing safety rule and was honored.
- The `heap_memory_percent` drirc arm used a synthetic
  `DRIRC_CONFIGDIR` (jw16 ships no drirc naming the option —
  `grep -l heap_memory_percent /usr/share/drirc.d/* /etc/drirc ~/.drirc`
  is empty), which is exactly the documented suppression path.
