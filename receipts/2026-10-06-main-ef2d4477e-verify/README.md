# Main verify: ef2d4477e — wheel, native suites, jwm1 Sub-vs-scalar decode smoke

Date: 2026-10-06. Actor: MainVerify (orchestrator commission: verify main before it
sits untested). Public receipt; the private twin is the lab notebook entry
`MainVerify/20261006T1155Z-fleet-mainverify-ef2d4477e-verify.md` with
`artifacts/MainVerify/ef2d4477e-verify/`.

## Pinned source
- Commit under test: `ef2d4477e3c36d417b975fc6af4142faa669651d` (origin/main;
  "omarchy: subgroup gather_qmm kernel for routed-MoE decode (2.0-2.1x)").
- Pre-merge base for A/B (only if a suite is red): `321b2788c`.
- Delta vs base: gather_qmm subgroup kernel (`gather_qmm_sub.comp` new, default ON,
  kill-switch `MLX_OMARCHY_GATHER_QMM_SUB=0`), dispatch in `primitives.cpp`,
  `compute.h` kernel ids, decode-shape numerics test in `test_matmul_family.cpp`,
  plus receipts/docs.
- Build tree verified: clean status, HEAD pinned by hash before any build; wheel
  byte-compared across two independent builds (see "Build provenance").

## Wheel (release-equivalent)
- Toolchain: OrbStack Arch ARM64 VM on the build Mac (placeholder `<build-vm>`),
  glslc 2026.3, gcc 16.1.1, Python 3.14.7, openblas headers,
  `CMAKE_BUILD_PARALLEL_LEVEL=8`, nice 19, one build at a time.
- Recipe: `DEV_RELEASE=1 MLX_OMARCHY_WHOLE_BUNDLE_DIR=<whole-encoder bundle>
  CMAKE_INCLUDE_PATH=/usr/include/openblas scripts/prepare-mlx.sh &&
  scripts/build-wheel.sh` (bundle manifest/program digests match the runtime pin).
- Wheel: `mlx_omarchy-0.32.4.dev202610061206+ef2d447-cp314-cp314-linux_aarch64.whl`
  sha256 `c56c9e00303cc2df7a43c8861e114777b40482b88a6762ff4791bbe4b4566feb`
  (416597619 B). Stamp `+ef2d447` matches the pinned commit.
- Byte-identity of rebuild #1 vs rebuild #2: wheels differ byte-wise (32 B; version
  timestamp + build-path embedding). Member-level: both `libmlx.so` carry zero
  REUSE_LAG symbols (`note_buffer_handle`, `live_by_handle_`) and each embeds
  exactly its own build directory — content-equivalent, both clean.
- SPIR-V identity: **VERIFIED, both match the pinned M2-toolchain references**:
  `matmul_f32_coopmat_qk.spv` =
  `bc6eb65bc8d580aaf5c3f28da51cfbc619aeb5bbb52ecde742597ac927d6b0f1`,
  `gather_qmm_sub_bf16.spv` =
  `fb361c3298914b2946da62bdd4b2337a8b49a4f5f5335a2dc8d7055f404c277f`

## Native suites (Apple M1, T8103 G13G, Omarchy Linux, placeholder `<m1-host>`)
Build: battery flags (`MLX_BUILD_TESTS=ON MLX_BUILD_OMARCHY=ON MLX_BUILD_METAL=OFF
BUILD_SHARED_LIBS=OFF CMAKE_BUILD_TYPE=Release`), glslc 2026.3 on the M1 host.
GPU lock `/tmp/m1-gpu.lock` (flock) around every GPU run; single process; 300 s
timeout per binary (host has no reset backstop).

| suite | cases | pass | fail | notes |
|---|---:|---:|---:|---|
| omarchy_fast_ops_tests | 43 | 43 | 0 | rc=0; 26 assertions inside two `doctest::may_fail(true)` cases (test_fast_ops.cpp:3506, :3701) reported as allowed failures; doctest Status SUCCESS |
| omarchy_matmul_family_tests | 25 | 23 | **2** | rc=1, 288 assertions failed: "gather qmm gathers experts with scales and biases" (96) + "gather qmm subgroup kernel matches scalar at decode shapes" (192). A/B vs 321b2788c: see below |
| omarchy_capability_sim_tests (6 profiles) | 42 (7x6) | 42 | 0 | rc=0 per profile: m1-honeykrisp-fork, m1-stock-no-coopmat, subgroup-size-64, small-shared-memory, no-cooperative-matrix, m1-g13-legacy |

Binary sha256s: capability_sim `dbccaae2a4bb0b20981e3aa0e62fe50279e910e13107e6e53cbe46f42bd7bfae`,
fast_ops `78027b43e1282ce9986f979a1468ca542d24624a68c05efd9ce2525984febdc9`,
matmul_family `c025a3868a5863650cd7571cf700a8765753f6d3c672bfc6f2955fe82f8e8467`.

### matmul_family A/B vs 321b2788c (same host, same procedure, GPU lock, 300 s cap)
| commit | cases | pass | fail | assertions |
|---|---:|---:|---:|---|
| 321b2788c (pre-merge base) | 24 | 24 | **0** | 82,942,463 / 0 failed — rc=0, Status SUCCESS |
| ef2d4477e (merge under test) | 25 | 23 | **2** | 82,942,723 - 288 failed — rc=1, Status FAILURE |

Binary: `omarchy_matmul_family_tests_321` sha256
`da92bdd7085d15aeb0839b30542bd9bc08294314aa969ecd5106d7c8d5fa837b`, built from a
verified-clean 321b2788c checkout (HEAD hash asserted, clean status asserted) with
the same flags. 12:32:51Z run.

**Verdict: the red is introduced by ef2d4477e.** Both failing cases are in the
gather_qmm family the merge touches: the merge modified the shared
`gather_qmm.comp` (not only the new `gather_qmm_sub.comp`) and `primitives.cpp`
dispatch; at ef2d4477e the pre-existing "gather qmm gathers experts with scales
and biases" case fails 96 assertions on this host where it passed at the base,
and the new "gather qmm subgroup kernel matches scalar at decode shapes" case
fails 192. Per the pre-agreed rule this is the revert trigger; the named tests
and counts above are the revert evidence.

## jwm1 decode smoke — DeepSeek-Coder-V2-Lite-Instruct-4bit
Model: mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit, local snapshot,
shard sizes byte-exact vs HF metadata (5316545548 + 3523543836 B). mlx-lm 0.31.3
patched per `scripts/apply-mlx-lm-patches.sh` (CONV_RING=0). Same wheel for both
arms; the A/B is the env flip `MLX_OMARCHY_GATHER_QMM_SUB=1` (default) vs `=0`,
one process per arm, 16 chained greedy steps, 3 warm, prompt fixed in the script.
Provenance printed per arm from the same venv.

**BLOCKED — hardware limit, not a gate result**: all attempts (12:25:09Z, 12:25:34Z,
12:26:41Z UTC, both arms) failed at model load with
`RuntimeError: [omarchy] vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY
(request=134217728 bytes; cache released)`. Host memory: MemTotal 15843696 kB
(~15.1 GiB), MemAvailable 13281696 kB idle; model weights 8840089384 B (8.8 GB)
do not fit this host's GPU budget. Provenance per arm verified against the exact
wheel (`verified: match`, dist 0.32.4.dev202610061206+ef2d447, version_match true).
Resolution needed from the orchestrator: run this smoke on a larger-M1 host, or
accept the supplementary result below as the M1 Sub-vs-scalar signal.

## Supplementary Sub-vs-scalar smoke on jwm1 (routed MoE that fits 16 GiB)
Model: mlx-community/OLMoE-1B-7B-0125-4bit (routed MoE, 64 experts, single shard
3892272579 B, byte-exact vs HF metadata), same wheel, same 16-token chained greedy
protocol, one process per arm, GPU lock held, provenance verified per arm.

| arm | ms/token median | token ids | greedy sha | text |
|---|---:|---|---|---|
| SUB=1 (default, Sub kernel) | 108.25 | 187 187 34 24720 310 247 1511 273 13732 326 310 18325 323 15896 285 3492 | e4364e17bbb9d501 | `A GPU is a type of processor that is optimized for graphics and video` (coherent) |
| SUB=0 (scalar kill-switch) | 1476.26 | identical | e4364e17bbb9d501 | identical |

Per-step ms (SUB=1): 107.53..108.80 (tight). Per-step ms (SUB=0): 1474.3..1479.99.
On this host the Sub kernel is 13.6x end-to-end decode throughput and produced
bit-identical top-1 tokens vs the scalar path at these shapes. This does not clear
the matmul_family red above — the suite's per-op fp64-anchored cases are the
numerics gate; the model smoke only shows the in-model decode path produced
coherent, arm-identical output.

## Compiled statement
| artifact | source commit | compiled | evidence |
|---|---|---|---|
| wheel `0.32.4.dev202610061206+ef2d447` (c56c9e00…) | ef2d4477e | **COMPILED** | HEAD-hash + clean-tree guarded build; libmlx member carries zero REUSE_LAG symbols; jwm1 install provenance verified=match |
| wheel `0.32.4.dev202610061215+ef2d447` (98f1fb25…) | ef2d4477e | **COMPILED** (independent rebuild) | second guarded build; member-symbol parity with wheel 1 |
| suites fast_ops (`78027b43…`) / capability_sim (`dbccaae2…`) | ef2d4477e | **COMPILED, RUN — green** | jwm1 logs, rc=0 |
| suite matmul_family (`c025a386…`) | ef2d4477e | **COMPILED, RUN — red** | jwm1 log, rc=1, 2 cases / 288 assertions |
| suite matmul_family_321 (`da92bdd7…`) | 321b2788c | **COMPILED, RUN — green** | jwm1 log, rc=0 |
| SPIR-V (both pinned shaders) | ef2d4477e | **COMPILED** | byte-identical to the pinned M2-toolchain references (glslc 2026.3 in the build VM) |
| smoke arms | wheel ef2d4477e | **RUN** | per-arm provenance verified=match |

The 321b2788c tree was built only for the matmul_family A/B (the only red
suite); fast_ops and capability_sim were not rebuilt at the base because both
are green at ef2d4477e and a green-vs-green A/B carries no information.

## Incidents
- 12:06-12:09Z build (first attempt, suites): a co-tenant automation edited the
  build clone's overlay mid-build (allocator/device/encoder; attributed to the
  REUSE_LAG lane, confirmed by its owner). Suite build failed on the mixed tree;
  the wheel (assembled before the edits) is byte-compared against the clean
  rebuild. Rebuild ran at a private path with clean-tree/HEAD guards and an
  overlay-dirty tripwire. Recorded in the notebook entry with timestamps.
- huggingface_hub `snapshot_download` returned DONE with only small files fetched
  (no error); shards were fetched by resumable curl and size-verified. Filed as a
  harness caveat; do not trust bare DONE from that path for multi-GB models.

## Acceptance per assignment
- Wheel from ef2d4477e, release-equivalent: **DONE** (two builds, stamp match,
  SPV pin match, provenance-verified install on the M1).
- Three named suites on ef2d4477e: **DONE — fast_ops green, capability_sim
  green (6/6 profiles), matmul_family RED**; 321b2788c A/B built and run:
  base green, merge red → revert trigger per the pre-agreed rule; tests named
  with per-case assertion counts.
- jwm1 DeepSeek Sub-vs-scalar smoke: **BLOCKED by host memory** (8.8 GB model
  vs ~15.1 GiB host; three deterministic OOM receipts). Supplementary
  routed-MoE smoke (OLMoE) delivered: SUB=1 108.25 ms/token vs SUB=0
  1476.26 ms/token (13.6x), identical 16 token ids, identical greedy sha,
  coherent text — a supplement, not the named deliverable.
- Compiled statement per sha: **DONE** (table above).
