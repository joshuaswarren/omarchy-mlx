# oMLX / TensorFold / MCDMA × omarchy-mlx — parity matrix (v1, 2026-10-04)

Owner lane: ParityMatrix. Living document — rows update as sibling lanes close them.

Pins (shallow clones, verified this session):
- **oMLX v0.7.0** — jundot/omlx tag `v0.7.0` = `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` (2026-10-01; latest release per upstream-watch digest 2026-10-04).
- **TensorFold main** — ashhart/TensorFold `609ca419abecebdc5a059498a613680bd3aa847f` (2026-10-03) — **identical to tag v0.6.5**, i.e. the commit already audited in `receipts/2026-10-04-tensorfold-audit/README.md`; that audit's file:line citations and runtime probes transfer unchanged.
- **MCDMA main** — ashhart/MCDMA `e672c14ff9fc7b38994caf73025cf1588b4de74e` (2026-09-29).
- omarchy-mlx base: origin/main `831f8fa03` (includes mx.fast.int8_matmul + translator extensions landed 2026-10-04).

Columns: feature | evidence (file:line at pinned commit) | needs Metal-only API? | runs on omarchy-mlx today | gap | owner lane.
Status vocabulary: `tested: <receipt>` / `untested` / `known-fail: <reason>` / `blocked: <gap>` / `n/a`.

---

## Proposed lane list and order of work (parity gaps only; no capacity claims)

| # | Work | Owner | Why this order |
|---|------|-------|----------------|
| 1 | **Platform-gate patch**: replace `mx.metal.is_available()` gates with a custom-kernel capability probe in omlx `omlx/patches/*`, `omlx/custom_kernels/qwen35_prefill/fast.py`, `omlx/engine/dflash.py`, `omlx/memory_monitor.py`, and TensorFold's 9 gate sites (draft branches already exist on the TensorFoldPort forks). Unblocks every gated JIT path + memory baseline accounting at once. | TensorFoldPort (pattern proven); OmlxLinux consumes | One shared gate unblocks ~25 omlx JIT sites + all TensorFold families; everything downstream depends on it |
| 2 | **Stack alignment**: omlx pins `mlx==0.32.2` (bundled kernel binaries ABI-coupled, `pyproject.toml:42-46`) + `mlx-lm` git `94cdcae1` (0.32 line) + `mlx-vlm@ea79808`, `mlx-audio@49596ac`, `mlx-embeddings@32981fa`, `transformers>=5.14,<5.18`. omarchy ships mlx 0.32.4-base + mlx-lm 0.31.3 (0.32-line patches already in `patches/mlx-lm-0.32/`, tool-call patch included). Probe/install each dep against the omarchy wheel. | OmlxLinux | Nothing serves until imports resolve; pure Python, no Metal |
| 3 | **Core server smoke on omarchy**: single + multi-model serving, continuous batching, paged SSD cache, embeddings/rerank, tool calling, admin UI, benchmark panel. Dev-box lavapipe first, then one real-GPU host via a ticketed window. | OmlxLinux; M2Lane verifies on real GPU | Proves rows A1–A21 in one pass; every row currently `untested` |
| 4 | **JIT kernel battery**: run each omlx patch JIT kernel + each TensorFold family kernel through the MSL→GLSL translator on the dev stack; classify runs / needs translator extension / exact-error (simdgroup_matrix, steel-MMA). | TensorFoldPort | Reuses the existing construct battery; converts `untested` kernel rows to tested/known-fail |
| 5 | **Native metallib families**: omarchy-native Vulkan equivalents or explicit fallback acceptance + perf ledger for glm_moe_dsa, minimax_m3, qwen35_prefill (classic + NAX), decode_fast SDPA, bonsai. NAX/MPP M5-only paths stay n/a on M1/M2 hosts. | TensorFoldPort (+ omarchy-mlx perf lanes for the ledger) | Depends on 1+4; biggest perf gap (GLM DSA prefill ~30x claim, README:120) |
| 6 | **mx.distributed → main, then oMLX cluster**: land GPU-stream collective primitives from `agent/omarchy-distributed`, then run oMLX pipeline shards / cluster roles over ring/TCP; 2-host ring when Main relays the second GPU host. JACCL/TB-RDMA and NCCL backends stay macOS/CUDA-only. | OmarchyDistributed; OmlxLinux consumes | Unblocks rows A22–A24 and TensorFold dflash drafter sharding |
| 7 | **Memory/wired-limit parity probe**: runtime probe `mx.set_wired_limit`, `device_info` keys (`max_recommended_working_set_size`), `get_active_memory` on omarchy; then dflash wired ownership + oq calibration sizing + memory-monitor baseline work with gates from #1. | OmlxLinux | Small; folds into #3's smoke |
| 8 | **H3 int8 dispatch finish**: quantize-kernel translator chase under asan (heap-corruption suspect already quarantined), then int8_linear/int8_swiglu dispatch end-to-end; M2 + macOS reference runs per the existing plan. | TensorFoldPort; M2Lane | Already in flight; mx.fast.int8_matmul landed (tests 2/2) |
| 9 | **MCDMA**: no omarchy work without an RDMA NIC (fleet has none). Record as conditional: Linux peer builds against stock libibverbs (x86-64/ARM64) the day a RoCE host appears. | Main (decision) | Hardware-gated, not software-gated |

---

## Metal-only API → omarchy equivalent

| Metal-only call / mechanism (sites) | omarchy-mlx equivalent | Status |
|---|---|---|
| `mx.metal.is_available()` — omlx: memory_monitor.py:36, engine/dflash.py:523, oq.py (via device_info), 12+ patch gates; custom_kernels/qwen35_prefill/fast.py:879; TensorFold: 9 gates (audit §1.2) | Returns **False** (`no_metal.cpp`; tested: receipts/2026-10-04-tensorfold-audit/README.md §4). Capability probe that should replace it: `mx.fast.metal_kernel` availability ORs `omarchy::is_available()` (`patches/mlx-omarchy-metal-kernel.patch:11-14`) | tested |
| `mx.metal.device_info()` / `mx.device_info()` — oq.py:4618 (exception → 1 GiB default), dflash.py:526 (`max_recommended_working_set_size`), TF kernels/device.py:21, threads.py:39, prefill_mm.py:215 | `gpu::device_info` provided (overlay device_info.cpp:18-40; memory + architecture strings) — `max_recommended_working_set_size` key needs a runtime probe | untested |
| `mx.set_wired_limit` / restore — dflash.py:530,544 (dflash owns BatchGenerator's lifecycle); TF 4 sites | Wired/cache-limit plumbing in omarchy allocator (audit §1.1) — but the `mx.metal.is_available()` gate at dflash.py:523 skips it entirely today | untested |
| `mx.get_active_memory` / `get_peak_memory` / `reset_peak_memory` — memory_monitor.py:313 (baseline), TF 10+ sites | Provided via omarchy allocator (audit §1.1) | untested at these sites |
| `mx.fast.metal_kernel` (MSL JIT) — omlx: 56 files matched `metal_kernel/MPP/xcrun`; JIT sites under omlx/patches/ + custom_kernels/qwen35_prefill/gdn.py:318,325,560,875; TensorFold: 30 call sites (audit §1) | MSL→GLSL translator + runtime SPIR-V (overlay custom_kernel.cpp). Supported: buffers/scalars/templates, threadgroup + barriers, subgroup reductions, output atomics, multi-output, shape metadata, bf16 buffers, fma. Landed on main 2026-10-04: `as_type` bitcast, `bfloat()` cast, subgroup-basic extension request, `_Pragma` strip, C-cast chain (scanner reverted pending asan re-land), 64-bit index→uint narrowing, `rint`→`roundEven`, MPP include/using strip. Rejected with exact compatibility errors: `simdgroup_matrix`/MMA bodies, textures, dynamic threadgroup, serialized scalars | tested (construct battery, audit §4 + notebook); per-omlx-kernel: untested |
| `simdgroup_matrix` / steel-MMA MSL — minimax m3 msa.py:344 (`_MSA_CSR_K1_STEEL_MMA`); TF qwen/dense simd_qmm `_MMA`, flash_next experts.py:322, stream_experts.py:68-100 | No equivalent; exact named error (tested). Hand-port analogue `matmul_coopmat` exists for rewrite-by-hand | tested (exact error) |
| Prebuilt metallib via `xcrun -sdk macosx metal` (.metal→.air→metallib, C++ ext loads it) — custom_kernels/{glm_moe_dsa,minimax_m3,qwen35_prefill,decode_fast,bonsai}/csrc/CMakeLists.txt:44-171 | **None.** Needs omarchy-native C++ Vulkan ops per family, or accepted silent generic-path fallback (README:120 documents fallback + `native_kernel_status()` surfacing, custom_kernels/__init__.py:17). Binaries ABI-coupled to mlx==0.32.2 (pyproject.toml:42-46) | known-fail on Linux (no Metal toolchain) |
| MPP `mpp::tensor_ops` int8 matmul2d (H3 W8A8, 128³ tiles, int32 accumulate) — drowzeys fork mlp_int8.py:20-60; omlx NAX kernels (nax_tiles.py:23, qwen35_*_nax.metal) | `mx.fast.int8_matmul` omarchy-native op **landed** (`patches/mlx-fast-int8-matmul.patch`; omarchy_int8_matmul_tests 2/2, int8_smoke PASS). Forks need a dispatch patch to call it | tested |
| NAX / M5 tensor-unit detection — custom_kernels/nax.py:1-9, qwen35_prefill NAX metallib (macOS SDK 26.2 gate, csrc CMakeLists:116-171) | n/a on omarchy M1/M2 hosts; NAX-gated dispatch degrades to classic path by design | n/a |
| `mx.distributed.{init,is_available,all_sum,all_gather,send,recv_like}` — omlx cluster/ (28 files), patches/*/shard+pipeline; TF z_lab dflash model_mlx.py:605 | GPU-stream omarchy primitives implemented on branch `agent/omarchy-distributed` (AllReduce/AllGather/Send/Recv/ReduceScatter; 2-rank harness; not yet on main). ring/TCP backends portable; `jaccl` (Apple TB RDMA) and `nccl` (CUDA) backends n/a | blocked (branch pending) |
| `platform.mac_ver()` / macOS-version gates — qwen35_prefill/fast.py:879-882 | Evaluates False on Linux → kernels off; graceful | tested (by contract) |
| Metal toolchain build (`OMLX_WITH_CUSTOM_KERNEL=1`, full Xcode) — README:84-108, setup.py | n/a on Linux; `native_kernel_status()` reports unavailable → families run generic paths slower | known-fail (by design, surfaced) |
| Swift/SwiftUI menubar app — apps/omlx-mac, README:261 | n/a; admin web UI is the Linux surface | known-fail (n/a) |

---

## A. oMLX v0.7.0 features

| Feature | Evidence | Metal-only? | omarchy today | Gap | Owner |
|---|---|---|---|---|---|
| A1 OpenAI-compatible server (chat/completions, completions, embeddings, rerank, models) | README.md:269-281; omlx/server.py; api/ | no | untested | None known beyond stack alignment (#2) | OmlxLinux |
| A2 Anthropic Messages API + adaptive thinking | README.md:269-281 (`/v1/messages`) | no | untested | — | OmlxLinux |
| A3 Streaming usage stats, SSE keep-alive, Claude Code context reporting | README.md:202-205 | no | untested | — | OmlxLinux |
| A4 Continuous batching via mlx-lm BatchGenerator | README.md:198; cache/factory.py:8-9; adapter/output_parser.py:404; cli.py:414 | no (mlx-lm) | untested | omlx pins mlx-lm git `94cdcae1` (0.32 line, pyproject.toml:49); omarchy wheel ships 0.31.3 + `patches/mlx-lm-0.32/` — alignment + BatchGenerator smoke needed | OmlxLinux |
| A5 Paged KV + prefix sharing + copy-on-write | cache/prefix_cache.py:176 (BlockAwarePrefixCache), :1288; cache/paged_cache.py:142 | no | untested | — | OmlxLinux |
| A6 SSD cold tier (safetensors offload, survives restart) | cache/paged_ssd_cache.py (PagedSSDCacheManager), README.md:187-197 | no | untested | — | OmlxLinux |
| A7 DFlash speculative decoding | engine/dflash.py; speculative/dflash_drafter.py; speculative/vlm_mtp.py | partial — wired-limit gate `mx.metal.is_available()` dflash.py:523 skips `mx.set_wired_limit` :530/:544; optional z_lab drafter uses `mx.distributed.all_sum` | untested | Gate patch (#1) + wired-limit probe (#7); drafter sharding blocked on mx.distributed (#6) | OmlxLinux + TensorFoldPort |
| A8 SpecPrefill (draft/planning/policy/target) | omlx/specprefill/{draft,planning,policy,target}.py | no | untested | — | OmlxLinux |
| A9 VLM serving (mlx-vlm@ea79808; multi-image, MiMo video+audio) | README.md:183-186; pyproject.toml:116; patches/mlx_vlm_{glm5_next,minimax_m3,qwen4_exp}_compat (vendored); adapter/output_parser.py:413-420 | no (Python dep) | untested | Pinned mlx-vlm commit vs omarchy mlx 0.32.4-base — import + processor probe needed (#2) | OmlxLinux |
| A10 OCR (DeepSeek-OCR, DOTS-OCR, GLM-OCR auto-detect) | README.md:183-186 (models table) | no | untested | — | OmlxLinux |
| A11 Audio STT/TTS/STS via mlx-audio@49596ac (DeepFilterNet, MossFormer2, SAMAudio, LFM2.5-Audio) | engine/sts.py:109-192; pyproject.toml:118 | no | untested | Dep probe (#2); needs no Metal beyond stock ops | OmlxLinux |
| A12 Native embeddings + rerank (BERT, BGE-M3, ModernBERT, XLM-R) | engine/embedding.py:101; engine/reranker.py:68; pyproject.toml:53 | no | untested | — | OmlxLinux |
| A13 Model profiles + per-model settings + alias + type override | README.md:216-227; model_profiles.py; model_settings.py | no | untested | — | OmlxLinux |
| A14 Multi-model serving: LRU eviction, pinning, per-model TTL, engine pool | README.md:206-215; engine_pool.py:323 (EnginePool), :2474; scheduler.py:1670 | no | untested | — | OmlxLinux |
| A15 Memory guard / process memory enforcement + monitor | process_memory_enforcer.py:376; cluster/memory_guard.py:456; memory_monitor.py:36,313-325 | partial — `HAS_MLX_METAL = mx.metal.is_available()` memory_monitor.py:36 gates `mx.get_active_memory()` baseline (:313) → baseline 0, KV-growth accounting degraded on omarchy | untested | Gate patch (#1): omarchy allocator provides `get_active_memory`, so probe can be True; then runtime probe (#7) | OmlxLinux |
| A16 Admin dashboard + chat + i18n + downloader + integrations | README.md:161-168,228-252; admin/routes.py:2392; admin/ web assets | no | untested | — | OmlxLinux |
| A17 Benchmark panel (PP/TG, partial prefix-hit) | README.md:253-260; admin/routes.py:84 (benchmark import); admin/bench_corpora/ | no | untested | — | OmlxLinux |
| A18 Tool calling (10 family formats), grammar/structured output, MCP | README.md:282-300; api/tool_calling.py:1993,2325; omlx/mcp/; mcp.example.json | no | untested | mlx-lm tool-call parser fix already in omarchy `patches/mlx-lm-0.32/` (MlxLm1904) | OmlxLinux |
| A19 oQ universal dynamic quantization (calibration-driven mixed precision) | docs/oQ_Quantization.md:1-6; oq.py | partial — `mx.metal.device_info()` fallback oq.py:4618 returns 1 GiB on omarchy (calibration memory sizing guess) | untested | device_info key probe (#7); otherwise pure mlx ops | OmlxLinux |
| A20 TurboQuant KV cache | turboquant_kv.py; cache/prefix_cache.py:4289,4673 (`mlx_vlm.turboquant` import) | no | untested | Lives in the pinned mlx-vlm fork — arrives with #2 | OmlxLinux |
| A21 MoE expert offload (stream from checkpoint safetensors, no converted copy) | docs/MoE_Expert_Offload.md:1-6; benchmarks/moe_offload_prefill_bench.py | no | untested | mmap slab reads are POSIX — expected portable | OmlxLinux |
| A22 Experimental multi-Mac pipeline shards (Ring/TB RDMA/JACCL), cluster dashboard, shard planning | README.md:169-182; docs/distributed-cluster.md:1; cluster/pipeline_compat.py:230-255; cluster/tensor_strategies.py:251; patches/deepseek_v4/deepseek_v4_model.py:2227-2244,2432; glm_moe_dsa_model.py:459-476; hy_v3_model.py:249-261; mimo_v2_model.py:474-487 | **yes — mx.distributed** (+ JACCL macOS) | blocked | mx.distributed GPU primitives on `agent/omarchy-distributed`, not main; ring/TCP path portable once landed | OmarchyDistributed |
| A23 Cluster consensus/telemetry/prefill-guard (all_sum voting, coordinated cancel) | cluster/collective_worker.py:14-18; telemetry.py:1186,1304,1582,1905; prefill_guard.py:201-214; runtime_optimizations.py:112,550-578; distributed.py:1011 | **yes — mx.distributed** | blocked | same as A22 | OmarchyDistributed |
| A24 RDMA stage transport + backend selection (jaccl/nccl probes) | cluster/rdma/stage_transport.py:281-335; cluster/backends.py:8; cluster/probe.py:109; cluster/autoconfigure.py:348; nccl_fabric_worker.py:87-110 | **yes** (jaccl = Apple TB; nccl = CUDA) | blocked / n/a | ring/TCP only on omarchy; jaccl+nccl stay n/a | OmarchyDistributed |
| A25 Native custom-kernel extensions, metallib families: glm_moe_dsa (GLM DSA prefill, ~30x claim), minimax_m3, qwen35_prefill (classic + NAX), decode_fast SDPA, bonsai | custom_kernels/*/csrc/CMakeLists.txt:44-171 (`xcrun metal`); custom_kernels/__init__.py:17 (native_kernel_status); README.md:84-120; bonsai rpath fix PR #4155 | **yes — Metal toolchain + metallib** | known-fail on Linux → silent generic fallback (by design, surfaced via /api/status) | omarchy-native Vulkan ops per family or accept fallback + perf ledger (#5) | TensorFoldPort |
| A26 JIT `mx.fast.metal_kernel` patch kernels (families): deepseek_v4 (hyper_connection.py:23,153; switch_layers.py:155; verify_qmv.py:300-334; wsdpa_attention.py:141,255), deepseek_v41 (activation.py:51; head.py:30; hyper_connection.py:35,91,150,231; kernels.py:336,534,606; packed_attention.py:355), gemma4_verify_kernel.py:346 (builds+runs probe), glm53_kda_prework.py:231, m5_gather_qmm.py:111, qwen35_fa256_attention.py:160, qwen35_gdn_chunked.py:47, qwen35_gdn_prework.py:1736, qwen35_moe_routed_decode.py:814, bailing_hybrid gated_delta (bailing_hybrid_model.py:407), glm_moe_dsa sparse_mla.py:15,229, glm5_next gated_delta.py:20,296, hc_prefill.py:361, minimax_m3 msa.py:344 (steel MMA), qwen4_exp hc_fused.py:727, hc_projection.py:385, qwen35_prefill/gdn.py:318,325,560,875 | per-site citations above | **yes — mx.fast.metal_kernel MSL** (+ `mx.metal.is_available()` gates at every site) | untested per-kernel; gates closed today (tested False); translator covers most constructs (tested), simdgroup/steel-MMA bodies known-fail exact-error | Gate patch (#1) then per-kernel battery (#4); steel-MMA sites need rewrite or exact-error acceptance | TensorFoldPort |
| A27 SDPA head-dim-256 O(L) tiled fast route + memory accounting | memory_monitor.py:43-66 (SDPA fallback modeling); patches/sdpa256_attention.py (route installer) | yes (Metal JIT route) | untested | Arrives with #4 | TensorFoldPort |
| A28 Laguna mlxfast port (bit-exactness-gated Swift→Python ports) | docs/laguna-mlxfast-port-correctness.md:1-7; omlx/patches/laguna/ | unclassified per-commit (ledger-gated) | untested | Each port lands only with token/bit-exactness + measured win — run ledger on omarchy during #3 | OmlxLinux |
| A29 macOS menubar app (SwiftUI, usage history, auto-update) | apps/omlx-mac; README.md:261-268; docs/usage-analytics.md | yes (Swift/macOS) | known-fail: n/a on Linux | Admin web UI covers monitoring on omarchy; usage_history.py is server-side | Main (accept n/a) |
| A30 ANE POC benches (qwen35_ane_{down_fused,down_output_split,gdn_split,prefill}) | benchmarks/qwen35_ane_*.py | macOS ANE experiments | n/a (macOS-only POCs, not shipped server features) | omarchy analog is the separate omarchy-ane lane — out of scope for these pins | Main (note) |
| A31 Web search tool + usage history (server-side) | websearch.py:80; usage_history.py | no | untested | — | OmlxLinux |

## B. TensorFold (main = v0.6.5, `609ca419`) features

Full audit: `receipts/2026-10-04-tensorfold-audit/README.md` (same commit; runtime probes included).

| Feature | Evidence | Metal-only? | omarchy today | Gap | Owner |
|---|---|---|---|---|---|
| B1 Five LLM kernel families (glm/flash, nemotron/lightning, qwen/dense, qwen/flash_next, gemma), 30 `mx.fast.metal_kernel` sites | audit §1, §1.4 (tf kernels/README.md:3-9; qwen/dense/v1/simd_qmm.py:339; glm/flash/v1/kernels.py:358) | **yes — MSL** | mixed: translator-allowed subset untested per-kernel; simdgroup_matrix subset tested exact-error | Per-kernel battery (#4); MMA sites stay exact-error or get coopmat rewrites | TensorFoldPort |
| B2 `mx.metal.is_available()` family gates (9 sites) | audit §1.2 (tf kernels/device.py:19; glm/flash/v1/{kda.py:285,fused.py:43,kernels.py:351,sparse_attention.py:82}; qwen/dense/v1/lane_gdn.py:192; threads.py:39; prefill_mm.py:215; gpu_sampling) | **yes** | tested known-fail: all gates closed (returns False) | Platform-gate patch; draft branches already on TensorFoldPort forks (#1) | TensorFoldPort |
| B3 Stock fast ops: rms_norm (36), sdpa (14, fused decode arm), rope (6) | audit §1.1 | no | tested (audit §4 + docs/compatibility.md:167,184-187,405) | f16 sdpa arm tolerance 0.01 documented | — |
| B4 Quantize paths: quantized_matmul (14), gather_qmm (6), dequantize (9) | audit §1.1 (tf deepseek/v4/rows.py:178; nemotron rows.py:416; qwen/flash_next/prefill_mm.py:275,370; glm/flash/v1/kernels.py:443) | no | tested (bits=4/8; shape-gated named errors; docs/compatibility.md:188-206) | mlx-lm Linear shape gating already handled | — |
| B5 mx.compile / async_eval / depends | audit §1.1 (41/4 sites) | no | tested (compiled bf16 tapes default; event graph) | — | — |
| B6 H3 video VAE conv3d | audit §1.1 (tfd families/h3/vae_video.py:215) | no | tested (general direct conv 1D/2D/3D) | — | — |
| B7 Weight IO (mx.load / save_safetensors) | audit §1.1 | no | tested (io-device patch) | — | — |
| B8 z_lab DFlash drafter sharding (`mx.distributed.all_sum`, optional) | audit §1.1 (tf drafters/vendor/z_lab_dflash/model_mlx.py:605) | **mx.distributed** | blocked | Needs #6; off critical path for single-host decode | OmarchyDistributed |
| B9 Device/memory APIs: device_info chip detect, active/peak memory, wired/cache limits | audit §1.1 (kernels/device.py:21; threads.py:39; prefill_mm.py:215; 10+ memory sites) | yes (metal namespace) | untested on omarchy at these exact sites | Probe with #7 | TensorFoldPort |
| B10 H3 int8 path: W8A8, MPP tensor_ops 128³ MMA, int32 accumulate, Int8MLP/Int8QKV | audit §2.1 (tfd kernels/minimax/h3/v1/mlp_int8.py:20-60,158-167; qkv_int8.py; families/h3/weights.py:73-115) | **yes — MPP MSL** | partial: `mx.fast.int8_matmul` landed + tested (omarchy_int8_matmul_tests 2/2); fork dispatch WIP behind quantize-kernel translator chase (heap-corruption suspect quarantined, asan) | Finish #8; MPP-MSL source keeps exact-error per contract | TensorFoldPort + M2Lane |
| B11 H3 wrapper platform gates (Darwin/arm64 die, 128 GB die, M5 gate) | audit §2.1 (h3w oneshot-setup.sh:33-38) | macOS-only gates | draft patches archived (TensorFoldPort branch; audit receipts dir on their branch) | Upstream drafts, Lead reviews | TensorFoldPort |
| B12 mlx-lm 0.32 module needs (gemma4_text, gated_delta, qwen3_next, qwen3_5, switch_layers, KVCache) | audit §3 | no | tested (all modules import + run on 0.31.3 under omarchy wheel; notebook 2026-10-04T21:2xZ) | Note: mlx-lm import runs mx.compile at import time → needs live accelerator (contract-consistent) | — |
| B13 Exactness contract (drafts verified against target's own sample; multi-row kernel must match its serial row) | audit §1.3 (tf engine/lane_engine.py:1,15; kernels/README.md:17-18) | no | design honored; end-to-end same-backend self-consistency on omarchy untested | Real decode run on M2 (#8/#3) | M2Lane |

## C. MCDMA (`e672c14f`) features

| Feature | Evidence | Metal-only? | omarchy today | Gap | Owner |
|---|---|---|---|---|---|
| C1 macOS RDMA driver + userspace verbs provider (ConnectX-5/7) | README.md:59-74; native/apple_provider.cpp:344 (sole Metal reference); native/*.cpp | macOS kernel/driver (Metal keepalive adjacent) | known-fail: n/a on Linux | None possible; omarchy hosts can only be peers | Main (accept n/a) |
| C2 Linux peer tool (bundled C, stock libibverbs, x86-64/ARM64; Strix Halo + Spark verified) | README.md:9-27; docs/linux-endpoints.md:3-23 | no | untested (buildable on omarchy; needs RoCE NIC + GID — fleet has none) | Conditional: build when an RDMA host exists | Main (decision #9) |
| C3 CLI (TypeScript) incl. Linux endpoint workflow | cli/package.json; cli/README.md | no | untested (installable) | None expected | — |
| C4 GPU-buffer transfer (Metal shared buffers on Mac; CUDA mapped host on Spark; no staging copy) | README.md:59-74 (integration section) | **yes** (Metal/CUDA memory) | known-fail: Vulkan mapped-memory RDMA registration unverified, no analog | Only relevant if an inference integration on omarchy ever needs RDMA of GPU memory — would need omarchy allocator + ICD (joshuaswarren/mesa-1) work; record as informational | Main (note) |
| C5 fabric-keepalive (continuous Metal GPU keepalive lowering RDMA latency) | README.md:36-58 | **yes — Metal (macOS GPU)** | known-fail: n/a | n/a | — |
| C6 Link daemon + KV handoff + vLLM connector (disaggregated prefill/decode) | docs/link-daemon.md; docs/kv-handoff.md; integrations/vllm/; README.md:59-74 | CPU/shared-memory + RDMA | untested (link daemon + KV handoff offline-tested upstream only; end-to-end run used scripts outside the repo) | Hardware-gated (C1/C2); informational for omarchy | OmarchyDistributed (informational) |

---

## Method and honesty notes

- Greps run at the pinned commits (2026-10-04): `mx\.metal\.` (25 files in omlx/omlx), `metal_kernel|MetalPerformance|xcrun` (56 files), `mx\.distributed|mlx\.distributed` (28 files), `mlx_vlm|mlx_audio|BatchGenerator` (128+ files); MCDMA full-tree Metal grep → 1 site. Evidence digest + SHA256SUMS: private notebook `artifacts/ParityMatrix/parity-matrix/`.
- No oMLX-on-omarchy run exists yet: every such row says `untested`, not "works".
- TensorFold rows reuse the same-commit audit receipt instead of re-deriving (its runtime probes were real runs on the omarchy wheel).
- mx.distributed status cited to the OmarchyDistributed branch/notebook, not assumed.
- No upstream PRs opened anywhere; no capacity/velocity claims made.
