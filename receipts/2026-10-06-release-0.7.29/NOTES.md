# 2026-10-07 — mlx-omarchy v0.7.29 (MoE subgroup kernels, CPU PD hold, direct cooperative GEMMs)

> DRAFT — not published. Publish only on Main's word after the gate runs
> (jwm1-class g-hold, jw16/M2 gates after the cluster window).

Release: v0.7.29, tag `v0.7.29` = `d86daf925` (origin/main at GO), annotated,
pushed at the cut sha before any gate ran. aarch64-only (the 0.7.28 decision
stands). Build host: the macstudio OrbStack VM `lsdc3build` with the pinned
toolchain prefix `/opt/m2-tc` (glslc 2026.3 / shaderc 1.4.357 — the VM's pacman
glslang 1.4.363 / shaderc 2026.4 were deliberately NOT used).

## Highlights (numbers quoted from committed receipts)

- **Routed-MoE decode: subgroup gather_qmm kernel + z-chunked dispatch.**
  GLM-4.5-Air-4bit per-layer decode 82.07 -> 9.71 ms at B=1 (8.5x), 8.1x at
  B=2, 9.5x at B=4; aggregate throughput 1.08x (B=2) / 1.38x (B=4) versus
  serial. DeepSeek-Coder-V2-Lite 70.9 vs 449.6 ms per token (6.3x), coherent
  output. Numerics judged against an fp64/ Metal reference; zero scalar
  dispatches left in the census at B=2.
  (`receipts/2026-10-07-moe-gather-qmm-perf/`)
- **Dense GEMM without staging (f16 and f32 direct cooperative matrices).**
  MLX `a @ b` fp16 4096^2: 0.598 -> 1.718 TFLOP/s (2.87x; standalone nn
  4096^3 2.88x, nt 2.44x). The cell moves from 0.26x to 0.75x of macOS Metal
  on the same die. 0 bit mismatches against the fp64 truth over all 16.7M
  outputs at 4096^3 plus 24 edge cases; identical output hashes with the
  direct route and `MLX_OMARCHY_NO_COOPMAT=1` through the wheel.
  (`receipts/2026-10-06-dense-f16-gemm-direct/`)
- **CPU PD hold (G13 non-C parts): the backend holds a PM-QoS CPU latency
  request only while GPU work is in flight.** matmul 4096 fp16 +26%
  (0.474 -> 0.598 TFLOP/s); real cells TTFT -1.3% (2B pf512) / -2.1% (2B d64)
  / -7.8% (4B d64) / -8.3% (9B d64), decode +0.6..0.8%; digests identical in
  every pair; idle power back to baseline (~3.51 vs 3.50 W). Ships with
  `packaging/udev/70-omarchy-mlx-cpu-dma-latency.rules` (group video 0660 +
  uaccess) staged by install.sh, and a package hook that reloads + triggers
  udev so no reboot is needed. The G13C (M1 Max) and G14 parts are untouched
  by default. `MLX_OMARCHY_CPU_PD_HOLD=0` opts out.
  (`receipts/2026-10-06-cpu-pd-hold/`)
- **Honeykrisp heap default on >32 GiB machines:** heap =
  max(50% MemTotal, MemTotal - 16 GiB), clamped at 60 GiB, set only when
  MemTotal > 32 GiB. On a 62.36 GiB M1 Max the usable heap went 30 ->
  46.36 GiB and Qwen3-32B-8bit loads. Smaller machines keep mesa's 50%
  default. (`receipts/2026-10-06-hk-sysmem-default/`)
- **Allocator fence fix (the NaN root-cause class):** fence-gated buffer
  recycling + transfer stamping, generation-gated quarantine, and the idle
  tick releasing the final generation once the drain settles
  (`6a925715a`, `06f089e0c`, `3fa344c3e`, `af0a3d5b1`, `2f0dbb2a7`).
- **GDN prefill recur32 (macOS-shape recurrence) is the default on G13
  non-C parts** — a TTFT lever (routes every prefill T >= 2 through the
  one-dispatch per-token recurrence). Order-only change: fp64-reference error
  equal or better, deterministic, long-prompt greedy tokens identical;
  `MLX_OMARCHY_GDN_RECUR32=0` restores the exact scan and the old pins.
  Batched GDN decode (B > 1) now takes the fused kernel with bit-identical
  per-row outputs. (`receipts/2026-10-06-gdn-recur32-default/`)
- **oMLX companion patches 0006-0009:** 0006 fuses the GDN decode conv chain
  in the MTP qwen35 model (+6.9% oMLX-server decode, identical completion
  texts; contract test red/green), 0007 Laguna rope_parameters tokenizer
  preflight guard, 0008 audio STS discovery for model_type-less checkpoints,
  0009 DeepFilterNet local-path subfolder rule. Kill switch
  `MLX_OMARCHY_OMLX_CONV_FUSE=0`.
- **Plus:** bool takes gather on the GPU (TakeBool / TakeMultiBool /
  GatherAxisBool) with host rank collapse; chunked composed SDPA
  output-write fixes; storage-buffer range refusal past the device limit;
  CPU quantized matmul rows accumulate in fp32; nine upstream MLX backports
  (`receipts/2026-10-06-mlx-backports/`); mlx-lm tool-call argument parsing
  backport.

## Assets (draft)

- `mlx_omarchy-0.32.4.dev202610070139+d86daf9-cp314-cp314-linux_aarch64.whl`
  sha256 `fe9c92d666a55c54e4dba46db28daaf34ebc1ebb184e16a7cd1063333f603849`
  (416,636,865 bytes; built on `lsdc3build`, 19 jobs, whole-encoder bundle
  staged from the published v0.7.28 wheel and pin-verified by
  `packaging/stage-whole-bundle.sh` before the build)
- `omarchy-mlx-vendor-wheels-v0.7.29-cp314-aarch64.tar`
  sha256 `39912d15400999aa66affde8a04c57efa07f9b0dfd86efd9b5c5c21e85655184`
  (461,209,600 bytes; 36 vendored wheels, verified twice by
  `packaging/verify-vendor.sh` — at generation and offline re-check)
  + `.sha256` (`1fe616c18e5447f9bedb5a5421eb3639d3ee246a0c125bf824d72cd1d0e42cd2`)
- `SHA256SUMS` (3-asset coverage, `sha256sum -c --ignore-missing` clean)
- DRAFT release created 2026-10-07 (untagged draft); draft verify
  `scripts/verify-release-assets.py v0.7.29 --platforms linux_aarch64`:
  `VERIFIED: every uploaded asset matches what the release claims`, rc=0
  (tag commit d86daf925 matches; version, feature strings, platform tag all
  PASS). Not published — publish waits on the gates.

### Build provenance and SPIR-V identity

- Toolchain: `/opt/m2-tc/usr/bin/glslc --version` -> glslc 2026.3, shaderc
  1.4.357.0 (enforced by `PATH` + `CMAKE_PREFIX_PATH`; the VM's pacman
  glslang 1.4.363 / shaderc 2026.4 were not used).
- `matmul_f32_coopmat_qk.spv` =
  `bc6eb65bc8d580aaf5c3f28da51cfbc619aeb5bbb52ecde742597ac927d6b0f1` ==
  the standing pin (shader source unchanged since the pin; byte-equality
  proves the toolchain identity).
- `gather_qmm_sub_bf16.spv` =
  `4a1db218a90de87add022f96d75f5a56cd5bcc96cb9e49caf15decf52389c44b` — the
  `fb361c32…` pin belongs to the shader BEFORE MoeLayer2's `1f66a03bb`
  (dtype-conditional PARAM_BYTES) and `b7e162060` (z-chunked dispatch):
  recompiling the three pre-change source states (`ef2d4477e`, `73ec9b9fb`,
  `b2bbaf998`) with the same pinned glslc reproduces
  `fb361c3298914b29…` exactly, and the tag's final source produces
  `4a1db218…`. The compile contract is
  `glslc -O --target-env=vulkan1.3 -DUSE_BF16=1 shaders/gather_qmm_sub.comp`.
- VM build env changes (all recorded, reversible): `pacman -Sy blas lapack
  lapacke python-pip` (build deps), `/usr/include/cblas.h ->
  /usr/include/openblas/cblas.h` symlink plus `CXXFLAGS=-I/usr/include/openblas`
  (Arch's openblas header layout vs mlx's `find_path(cblas.h /usr/include)`);
  full build log `/var/tmp/rel0729-build.log` on `lsdc3build`.

## Known issues

- The mlx-lm serve patches need `patch`(1). Omarchy's base install ships it
  (verified 2026-10-07 on stock x86 Omarchy and a fresh Omarchy M+ image,
  both `patch 2.8-1`); pacman users get it through the PKGBUILD's
  `makedepends`. The installer and `apply-mlx-lm-patches.sh` now fail with
  an explicit "patch is not installed" message instead of a misleading
  "mlx-lm version mismatch" (fixed on main after v0.7.29 was tagged; the
  v0.7.29 tarball keeps the old script).

- **Batched prefill of GDN/linear-attention models: rows beyond the first
  can produce wrong values (OPEN, pre-existing).** On the mlx-lm 0.31.3
  batch path (BatchGenerator) with omarchy's batched prefill/decode ops, a
  batch of four IDENTICAL prompts shows rows 1-3 diverging from step 0
  (3.31/3.75/4.63 nats; first differing module `layer0.linear_attn` at
  T=16) while row 0 and every single-prompt run are correct. Present in
  v0.7.28 and v0.7.29 alike (bit-identical results on both tags — not a
  v0.7.29 regression); cause under investigation (GDN prefill path at B>1).
  **oMLX users: continuous batching (completion_batch_size 8, the default)
  on GDN-family models (Qwen3.5/3.8) can therefore serve wrong rows in
  prefill; single-request serving is unaffected.** A per-server workaround
  (completion_batch_size 1) is being confirmed and will be added here.

- SDPA fused-VJP value defects remain open (qL=1 maskless dk/dv zero; B=1,
  kL=5 also fails).
- `compile()` of the sin*cos tape segfaults on the G13 build host; that leg
  stays out of the suite.
- Kokoro voice output remains unqualified.
- Laguna (GLM) prefill beyond ~480 tokens is slow (~7 tok/s) and can hang;
  evidence chain queued. Decode is unaffected.
- The routed-MoE subgroup kernel covers the proven bf16 layout class; f16
  scales, group sizes 32/128, 8-bit, and mxfp4/nvfp4 stay on the scalar
  kernel.

## Verification gates (pending; this draft precedes them)

- g-hold on the T8103/G13G host (through the power lane): udev + live fd test
  on the release wheel; decode digests bit-identical to the standing pins.
- jw16/M2 gate battery after the GLM cluster window frees them.
- `scripts/verify-release-assets.py v0.7.29 --platforms linux_aarch64` on the
  draft assets.
