# BM32 BF16 projection on the 16-inch M1 Max

| Kernel | Status | Max difference | Tested commit | Date (UTC) |
|---|---|---:|---|---|
| `nemotron_bm32` | PASS, bit-exact | 0 | `97bcab6a325c0a582622c6bed84bc00bea68e7c0` | 2026-11-21 |

Coverage for #523. The tested tree includes the probe from #530 and is identical to the #523 tree at `2c883cf316b57fb9d30c4804a05b3dbbbfd9bdf5`.

Device: Apple M1 Max (G13C C0), Honeykrisp, Mesa `26.3.0-devel (git-69bde15340)`. Kernel: `7.1.12-2-20261103.2-lab-ARCH`. `VK_DRIVER_FILES` selected only the Apple Vulkan driver. The same Release build supplied the probe and the statically linked timing binary.

`omarchy_matmul_family_tests -tc='*BM32*'` passed both cases and all 8,227 assertions in each of three runs. For the 32×4096×14336 BF16 projection, the dense call emitted one BM32 dispatch and the four chunks emitted four. The probe compared both outputs to an integer round-to-nearest-even oracle.

| Timed run | Flag off (ms) | Flag on (ms) |
|---|---:|---:|
| 1 | 1.6585 | 2.2255 |
| 2 | 1.4650 | 2.2245 |
| 3 | 1.4380 | 2.2280 |
| Median | 1.4650 | 2.2255 |

Each cell is the median of eight synchronized calls after discarding the first call of a fresh process. Every off sample recorded kernel 413; every on sample recorded BM32 kernel 695. BM32 was 52% slower for this dense shape on this GPU. Keep it opt-in. This is an isolated projection, not a model benchmark. The chunked probe was not timed.

The timing source and raw probe and timing logs are included. `SHA256SUMS` covers the evidence files.
