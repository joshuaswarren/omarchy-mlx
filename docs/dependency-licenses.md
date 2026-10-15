# Dependency licenses

mlx-omarchy is MIT. Every external dependency this project consumes at
build time, run time, or reference time is listed here with its license
and how it reaches the product. Nothing under a copyleft license is
linked or vendored.

## Build-time and runtime dependencies

| Dependency | License | How it arrives | Notes |
|---|---|---|---|
| MLX (upstream base) | MIT | source archive pinned by `mlx.lock`, patched by `patches/*.patch` | the overlay carries the Omarchy backend; upstream files keep their notices |
| {fmt} | MIT (span-format) | FetchContent at configure time | header-only usage |
| doctest | MIT | FetchContent at configure time | tests only |
| nlohmann/json | MIT | FetchContent at configure time | tests and Python tools |
| Vulkan SDK (headers + loader) | Apache-2.0 | system package | dlopened at runtime; headers only at compile time |
| GNUstep (libobjc2, gnustep-base, gnustep-make) | MIT-family | built by the pinned compiler's `scripts/bootstrap-linux-toolchain.sh` | used only to build mil-hwx-compiler; never shipped inside mlx-omarchy |
| protobuf | BSD-3-Clause | system package | the vendored Core ML schema bindings are generated from Apple's proto sources |
| Python (tooling) | PSF-2.0 | system | host-only front-end tools |
| mlx-lm Muse Spark text model (closed PR 1975, `117b84cae44a`) | MIT (Apple Inc.) | vendored unchanged: `serve/mlx_omarchy_muse/muse_spark.py` | licence text in `LICENSES/mlx-lm.txt`; repository checkout only |

## Project-owned external repositories (consumed, not vendored)

| Repository | License | Pin | Consumption |
|---|---|---|---|
| `joshuaswarren/mil-hwx-compiler` | MIT (© 2026 maderix) | `ane-compiler.lock` — commit `b61de46…`, release `ane-parity-b61de46` | compiled to a standalone binary under `.work/`; no compiler internals are copied into mlx-omarchy; `DISCLAIMER.md` travels with the compiler, not with us |
| `joshuaswarren/omarchy-ane` | MIT (© 2022 Eileen Yoon) | out-of-tree kernel module on the target host; `libane` dlopened by the worker CLI | the kernel driver and `libane` stay in their repository; mlx-omarchy links nothing from it at build time |

## Reference and test assets

| Asset | License | Where pinned |
|---|---|---|
| Parakeet Core ML model (`mweinbach1/parakeet-tdt-0.6b-v3-coreml`) | CC-BY-4.0 | `overlay/tools/coreml/parakeet-reference.lock` |
| Parakeet reference implementation (`mweinbach/parakeet-coreml-swift`) | Apache-2.0 | same lock (`reference_commit`) |
| Core ML protobuf schema (Apple coremltools) | BSD-3-Clause + Apple protos | `overlay/tools/coreml/schema/` (vendored bindings + `.proto`; provenance in `schema/VENDORED.json`, `schema/LICENSE.txt`) |
| LibriSpeech audio fixture | CC-BY-4.0 (upstream LibriSpeech) | same reference lock (`audio` entry, license field recorded) |

## Distribution obligations

- MIT and BSD notices travel in `LICENSES/` and inside the vendored
  trees; releases reproduce them (the wheel carries `LICENSES/`).
- The CC-BY-4.0 reference model is downloaded on demand by
  `mlx-omarchy-parakeet download` and cached per user; it is never
  embedded in a release artifact.
- The kernel module and compiler binaries are built on the target or
  developer host respectively and are not redistributed by mlx-omarchy.
