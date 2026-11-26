# M1 decode gap plan — 2026-09-06

## Gap (pinned)

Same M1 silicon, Qwen2.5-0.5B 4-bit, greedy, fixed lengths, median of 5.

| Workload | Native Metal | Linux Honeykrisp | Gap |
|---|---:|---:|---:|
| short 30/32 decode | **150.8 tok/s** (6.6 ms/step) | **31.1 tok/s** (~32.2 ms/step) | **~4.85× / ~25.6 ms/step** |
| long 262/128 decode | 146.6 | 28.3 | ~5.2× |
| longctx 1053/32 decode | 140.3 | 23.5 | ~6.0× |
| short prefill | 294 tok/s | 79 | ~3.7× |
| long prefill | 1213 | 188 | ~6.5× |
| longctx prefill | 1838 | 198 | ~9.3× |

Sources: `receipts/native-baseline-2026-09-06/native-2026-09-06-summary.json` (native mean/token 6.6–7.1 ms); `README.md` Performance table + `receipts/2026-09-04-m1-performance-gates.md` (Linux 31.07 / 28.32 / 23.53 decode; prefill 79 / 188 / 198 with QMM tile ON).

Target: close as much of the **~25.6 ms/step** short-decode residual as code can move; separate Honeykrisp/Mesa ceilings from our costs.

---

## Where ~32 ms/step goes (current structure)

### Already shipped (do not re-propose)

| Change | Result | Receipt |
|---|---|---|
| 4-slot CB ring + descriptor pool cache | Removed per-record join (~372 ms/tok single-core) and per-dispatch pool create (160–260 µs → ~1.7 µs median on llvmpipe) | `2026-09-03-submission-ring-descriptor-cache.md` |
| Deep batching 16–100 ops/submit | **4.5× slower** — deleted. MLX `MAX_ACTIVE_TASKS` + finalize turns batch flush into stop-and-wait | same |
| Watchdog sleep-poll | v0.3.4: 12.5 → 0.21 tok/s; fixed v0.3.5 | `2026-09-04-v0.3.4-decode-regression.md` |
| `qmm_vec` GEMV (m==1) | **+37%** 4-bit (12.83 → 17.61); subgroupAdd **parity** with shared tree | `2026-09-04-qmm-gemv-subgroup-m1.md` |
| QMM prefill tile | Prefill **2.2–5.5×**; decode unchanged | `2026-09-04-m1-performance-gates.md` |
| Fused RoPE | Dispatches 1868 → 715 (−62%) | CONTRIBUTOR-GUIDE; FastRope* in `primitives.cpp` |
| SDPA f16 scores path | −120 disp/tok; **+2.1%** only (12.56 → 12.82) | `2026-09-04-sdpa-f16-scores-m1.md` |
| SwiGLU `FusedChain` | 585 → 537 disp; **~+1%** decode | `2026-09-04-swiglu-fused-chain.md` |
| KV eval densifier removed | Correctness; no standalone speed claim | `2026-09-04-kv-state-views.md` |
| Dispatcher-wakeup poll | noise | CONTRIBUTOR-GUIDE stood-down |
| Compile-forward alone | −27–39% | stood-down |
| Generic tape fusion 7.3% ceiling | too narrow | stood-down |
| bf16 RoPE/SDPA direct gates | identity fail — stay off | `2026-09-04-m1-performance-gates.md` |

### Decomposition of remaining ~32 ms/step

Evidence base is composite: early full GPU profile under 1-core host contamination (`2026-09-02-gpu-profile-decode.md`), plus later wall-clock A/Bs on 8-core that show **kernel time, not dispatch count, is now the wall**.

| Bucket | Evidence | Est. share of 32 ms | Ours vs Mesa |
|---|---|---|---|
| **A. Kernel arithmetic (qmm_vec, MatmulF16, Softmax, RMSNorm, RoPE, EW)** | 2026-09-02: GPU busy ~35.7 ms/tok @ 1585 disp, kernel p50 ~17 µs floor. Post-GEMV + SDPA f16 + RoPE, wall ~32 ms at ~585–715 disp. SDPA −120 disp → only +2%; SwiGLU −48 → +1% ⇒ residual **not** host-record dominated. Native 6.6 ms total ⇒ Linux GPU path still ~4–5× Metal for same math. | **14–20 ms** | Mostly **our shaders** (tiling, unroll, register pressure). Some Honeykrisp AGX limits (no coop matrix, register-bound — llama.cpp disables unroll here). |
| **B. Blanket pre+post barriers** | Intra-gap p50 **35 µs**, 71–75% consecutive pairs **disjoint** compute bindings. ~9–12 ms/tok barrier tax at old disp count; scales with disp → **~4–7 ms** at 585–715. | **4–7 ms** | **Our encoder** (`dispatch_compute` always barriers). |
| **C. Remaining dispatch structure** (UpdateDescriptorSets, bind, push, submit-per-finalize, ring slot wait) | Ring+cache landed; deep batch deleted. `kBatchNodeBudget=100` batches within graph; finalize flushes at throttle. SwiGLU residual ~11 µs/disp wall. | **3–6 ms** | Mix: our finalize/throttle; Honeykrisp **submit latency** + **single compute queue**. |
| **D. Per-QMM scales\|biases `copy_buffer`×2** | `QuantizedMatmul::eval_gpu` builds combined buffer every call | **1–2 ms** | **Ours** |
| **E. Python / mlx-lm / scheduler throttle** | Profile OTHER ~11% on llvmpipe | **1–3 ms** | Upstream shape |
| **F. Allocator invalidate on join** | `join_last_completion` | **<1 ms** steady-state | Ours, small |

**Headline:** After ring/cache/GEMV/RoPE/SDPA-f16, dominant remaining gap is **(A) kernel work + (B) full barriers**, not one-dispatch-per-submit. Fresh census on current main is step 0.

### Honeykrisp / Mesa ceilings

- Single compute queue; no multi-queue overlap
- No coop matrix / int dot / native bf16 on M1 VK
- Register-bound AGX (unroll hurts)
- Driver submit + timeline cost per vkQueueSubmit
- subgroupAdd parity with shared tree — not a free win
- 2048-token single-eval wedge — separate defect

---

## Ranked plan

### 0. Fresh M1 diagnostics census (gate)
Diagnostics wheel + GPU_PROFILE + profile_generate/analyze on current main, 8 cores. Expected: 0 ms; unlocks ranking.

### 1. Dependency-gated barriers ★ TOP-3 — **4–7 ms/step**
Evidence: 2026-09-02 intra-gap 35 µs, 71–75% disjoint. Change: range tracker in CommandEncoder; barrier only on overlap. Risk HIGH (silent wrong values). Env-gated.

### 2. qmm_vec kernel quality ★ TOP-3 — **3–6 ms first slice** (8–14 if GPU≥half wall)
Evidence: native 6.6 vs 32 ms; GEMV +37% already; subgroup body not lever. Change: no-unroll, COLUMNS_PER_GROUP, dequant packing in qmm_vec.comp. Isolated A/Bs.

### 3. Descriptor freelist + scheduler-aware commit ★ TOP-3 — **2–5 ms/step**
Evidence: deep batch DELETED; UpdateDescriptorSets still per disp. Change: freelist sets; flush when scheduler would block — never fixed 16–100 op batch. Risk MEDIUM lifetimes.

### 4. QMM scales|biases pack-once — **1–2 ms**
primitives.cpp ~5840: two copy_buffer per affine qmm.

### 5. FusedChain default-on — **≤0.5 ms** today (585→537, +1%)

### 6. Real fused tape codegen / mlx-lm compile — large iff codegen exists; compile-alone measured regression

### 7. Kernel-vs-Metal microbenches — feeds item 2

### Rejected re-runs
Deep batching; subgroupAdd primary; dispatcher poll; compile-forward alone; bf16 direct gates; generic tape fusion 7%.

---

## Top-3 sketches

### TOP-1 Dependency-gated barriers
**Files:** encoder.h/cpp (`dispatch_compute`, `copy_buffer`, `fill_buffer`, `submit`); gpu_profiler barrier counters; runtime tests.
**Functions:** CommandEncoder::dispatch_compute — replace unconditional pre/post barriers with overlap test against unsynced_ write ranges; mark writes after dispatch; clear on submit. Keep MLX_OMARCHY_TAPE_FULL_BARRIERS force-full.
**Validate:** llvmpipe battery; M1 batteries; digests 7fd25a869ff21678 / 254d73fd93164b98 / 7da83f06ec9f001d; FULL_BARRIERS=1 matches gated digests; profile barriers↓ busy↑; bench_decode 5× median. Stop on any silent drift.

### TOP-2 qmm_vec quality
**Files:** shaders/qmm_vec.comp, qmm_tile.comp; primitives.cpp QuantizedMatmul::eval_gpu (m==1 ~5917–5945); CMakeLists shader defines; tools/subgroup-bench extend.
**Functions:** variant select via MLX_OMARCHY_QMM_VEC_VARIANT; microbench GPU ts before wall A/B.
**Validate:** matmul_family_tests; qmm ulp compare; greedy digest; bench_decode short+long; no prefill/decode cross-regress. Success ≥3 ms/step kernel total↓.

### TOP-3 Descriptor freelist + scheduler-aware commit
**Files:** encoder acquire_descriptor_set; eval.cpp finalize; event.cpp flush contracts.
**Functions:** freelist VkDescriptorSet by layout, Update only; do NOT accumulate across wait_for_one (deleted deep batch). First PR freelist only.
**Validate:** runtime stress 40/40; profile host record/submits/tok; digests; watchdog still fires on 2048 wedge. Success ≥2 ms or host share −30% rel.

---

## Sequence
0 census → 1 barriers (4–7 ms) → 2 qmm_vec (3–6 ms) → 3 freelist (1–3 ms) → 4 scales pack (1–2 ms) → re-census.
Optimistic stack 12–20 ms off → ~50–80 tok/s short still short of 150. Rest needs fused codegen or driver.

## Measurement
mlx_provenance verified=match; bench_decode.py pinned 64 tok median-5; diagnostics profile_generate + profile_analyze. Distinct wheel stamps; clear cmake cache.
