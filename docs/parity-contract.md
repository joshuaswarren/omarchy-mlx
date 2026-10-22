# Linux MLX parity contract and test plan

This contract defines correctness and throughput acceptance for MLX on the Apple GPU under Linux against native macOS MLX. It covers dense prefill, dense decode and sparse-MoE execution. A result qualifies only for the recorded model, device, software and workload.

## 1. Scope and authority

Keep `import mlx.core as mx` and `mx.gpu` unchanged. Release inference must execute tensor operations on Vulkan, with no CPU tensor fallback. CPU scheduling, tokenization and buffer transport remain allowed. Disable ANE placement for this GPU comparison; qualifying ANE regions requires the separate crossover contract.

Use this document for the engineering comparison below. It does not replace the [numerics gate](numerics-gate.md), [token identity policy](parity-id-policy.md), [compatibility release requirements](plans/2026-08-29-mlx-omarchy-ane-compatibility-plan.md#success-criteria), or the standing chip qualification batteries in [AGENTS.md](../AGENTS.md).

The engineering comparison requires at least five independent runs per arm and cell. The existing release procedure still requires five warmup runs and 30 measured repetitions, plus the 100-request stability gate. A five-run engineering result cannot qualify a release by itself.

## 2. Source identities and comparison boundaries

Use these immutable source pins to identify each comparison. Freeze the actual parent, candidate and integration commits before capture.

| Work | Source pin | Change boundary | Required discriminator |
| --- | --- | --- | --- |
| Common implementation base | `94f5ac8ca685173e801ec1023df5de9a17465bc4` | Existing Vulkan backend and model routes | Same-head control and independent numerical oracle |
| Dense prefill, [PR #257](https://github.com/joshuaswarren/omarchy-mlx/pull/257) | `b07648943a19f8d00c6751ea853cd34b7c39f6b2` | Paired subgroup output drains in `qmm_coopmat.comp`; shared index helpers and shader dependencies | Real cooperative-matrix output, scratch ownership and barriers on the target driver |
| Dense decode baseline | `94f5ac8ca685173e801ec1023df5de9a17465bc4` | Quantized projections, attention and recurrent decode routes | Each decode candidate requires its own source pin, changed-route trace and baseline comparison |
| Sparse MoE, [PR #256](https://github.com/joshuaswarren/omarchy-mlx/pull/256) | `086b85104555bffa14b0e49ccd072c4455e3ab26` | Direct scales, biases and routing buffers for tiled gathered Q4 prefill | Actual `gather_qmm` dispatch, original-buffer offsets/lifetimes and removal of parameter staging |

The upstream MLX numerical reference is version 0.32.3 at `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`, matching [mlx.lock](../mlx.lock). Record a different reference as a new comparison, never as the same baseline. Pin the same mlx-lm source, model code and project patch set on both platforms, except backend selection. Freeze their digests before capture.

### 2.1 Linux qualification build

Use a CPU-disabled wheel for Linux model qualification and timing. Keep a separate CPU-enabled binary for CPU oracle and test work. A CPU-enabled oracle binary cannot supply qualification timings.

The standard `scripts/build-wheel.sh` sets `MLX_BUILD_CPU=ON` and replaces external `CMAKE_ARGS`. Build the qualification wheel directly from the prepared source instead. Use a clean checkout per arm, an unused work directory, and a provisioned build environment with pinned dependencies:

```sh
export MLX_OMARCHY_WORK_DIR="$PWD/.work/parity-nocpu-$(git rev-parse --short=7 HEAD)"
export MLX_OMARCHY_LOCAL_VERSION="nocpu.$(git rev-parse --short=7 HEAD)"
export DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=2
scripts/prepare-mlx.sh
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF"
export CMAKE_ARGS="$CMAKE_ARGS -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF"
export CMAKE_ARGS="$CMAKE_ARGS -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF"
export CMAKE_ARGS="$CMAKE_ARGS -DMLX_OMARCHY_GPU_PROFILING=OFF"
python3 -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$MLX_OMARCHY_WORK_DIR/wheels" "$MLX_OMARCHY_WORK_DIR/mlx"
```

This recipe selects the pinned upstream `mlx/backend/no_cpu/` tensor implementations through `MLX_BUILD_CPU=OFF`. Preserve its CMake cache, configure/build log and compile commands. Require `no_cpu/primitives.cpp` and `no_cpu/device_info.cpp`, rather than the CPU tensor implementation sources. Host scheduling files such as `backend/cpu/eval.cpp` and `encoder.cpp` remain allowed. Match the built and loaded library hashes and wheel RECORD before running the controls.

Run controls in separate processes using that exact wheel. First, evaluate a nontrivial GPU addition and require the expected values and positive Vulkan dispatch counts. In the CPU negative control, create a new explicit CPU stream with `mx.new_stream(mx.cpu)`, then evaluate `mx.add(x, y, stream=cpu_stream)` on two-element float32 inputs. Require the pinned no-CPU error `Add has no CPU implementation.`; an import error, stream-index error, unrelated exception or successful result fails the control. Use a new stream rather than the default CPU stream, because the no-CPU backend reports zero CPU devices.

Record the CPU-disabled build identity, loaded library identity, exact control commands, output and exit status with P7. GPU-only counters cannot establish CPU exclusion. The implicit-default-device refusal patch cannot replace this explicit CPU control. Keep the GPU profiling build option off for timing; diagnostic captures and CPU-enabled oracle runs retain separate identities. Apply the full packaging gates separately before publishing a release wheel.

### 2.2 Model identities

| Model | Immutable snapshot | Required format |
| --- | --- | --- |
| `SiddhJagani/Qwen3.8-2B-mlx-4Bit` | `0867d98bfb174b042d88461c0e7c97b86b34b381` | Affine 4-bit, group 64 |
| `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` | `55ac8c89261109b36c04371cd3f479a4594208c8` | Affine 4-bit, group 64; 128 routed experts, six selected per token |

Record SHA-256 for every weight shard, configuration, tokenizer file and index file. Repository names, snapshot directory names and quantization labels alone do not establish identical tensors. Reject a mismatch instead of substituting another model or quantization.

Retain the original fixed 32-token Qwen compatibility gate separately. Its source plan names a GGUF artifact with SHA-256 `4aa0fb13c431514262f259d420ecc95a8714df58ac2a2384514e20b93983f0ff`. A safetensors snapshot is not that artifact. Bind the legacy oracle through an audited tensor/conversion manifest and its exact prompt IDs; do not replace its expected tokens with the engineering model's output.

## 3. Operation acceptance matrix

Record source, software-driver, physical-device and model checks separately. Qualification requires every mandatory check at its specified execution level.

| Required area | Source and existing test anchors | Required checks |
| --- | --- | --- |
| Dense quantized prefill | `overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp`; `overlay/tests/omarchy/test_matmul_family.cpp` cooperative-matrix and QMM tile cases; #257 `tests/qmm_drain_check.cpp` | Compile all 40 declared shader variants. Execute every variant admitted by the target's capabilities, including M16/M32, both subgroup scratch regions, partial rows/columns and repeated tiles. Unavailable capabilities must select their documented fallback or named refusal. A CPU index check cannot establish cooperative stores or GPU memory ordering. Verify actual kernel IDs. |
| Dense decode | `overlay/mlx/backend/omarchy/primitives.cpp` QMM and GDN selectors; `test_matmul_family.cpp`; GDN cases in `overlay/tests/omarchy/` | Capture the routes used by the pinned 2B model. Cover single-token quantized projections, normalization, recurrent state, convolution, attention/KV transitions and compiled/eager boundaries. Bind each improvement claim to a distinct candidate build. |
| Router and top-k | `primitives.cpp` `ArgPartition::eval_gpu`, `dispatch_sort`; mlx-lm `nemotron_h.py` `group_expert_select` | Test top-six selection from 128 scores, correction bias, normalization, ties, repeated selections and association between each expert ID and its score. The existing full-sort implementation does extra ordering work; measure its cost before choosing a selection optimization. |
| Quantized expert matrices | `dispatch_gather_qmm`, `GatherQMM::eval_gpu`; `gather_qmm_sub.comp`, `gather_qmm_tile.comp`; gathered-matmul cases in `test_matmul_family.cpp` | Cover both selected expert projections, K/N = 2688/1856 and 1856/2688, bf16/fp16 parameters, all selected expert IDs and captured model activations. Validate packed nibbles and scales/biases against independently dequantized operands. |
| Tile admission and fallback | #256 `primitives.cpp:3349-3420`; `test_gather_qmm_tile.py` and `test_gather_qmm_tile_standalone.cpp` | Exercise sorted/unsorted routes, non-identity and repeated input indices, absent experts, partial 32-row tiles and multiple experts in a tile. Use the isolated admission and offset matrix below. Confirm direct admission with seven bindings, per-row fallback below seven bindings and with `DIRECT=0`, and insufficient shared memory. |
| Unquantized expert gather | `GatherMM::eval_gpu` in `primitives.cpp` | Require its named refusal above 65,535 routed batches. Apply affine-Q4 expert qualification to `GatherQMM` separately. |
| Hybrid recurrent execution | mlx-lm `nemotron_h.py`, `ssm.py` and cache implementation; convolution, scan/reduction and compiled-tape tests | Establish convolution and SSM state oracles, prefill/chunk/decode equivalence, fresh-cache reset and state reuse across 128 decode steps. Sparse-matrix coverage alone cannot qualify this hybrid model. |
| Execution safety | Runtime, error-contract, copy-offset and capability-simulation suites; [differential tools](differential-harness.md) | Prove the trace detector is live, required routes executed, CPU tensor dispatch is impossible, and a new process can reopen the device. Refuse unsupported operations with operation/dtype/shape. |

For tile admission, fix affine Q4/group-64, transposed weights, `m=1`, sorted expert IDs, K/N multiples of 64, seven bindings, direct buffers enabled, and sufficient shared memory. Set each expert count from the leading weight dimensions; routed rows are synthetic gather entries, not token counts. Vary only the boundary under test:

| Expert count | Routed rows | Expected route |
| --- | --- | --- |
| 8 | 63 / 64 / 65 | Per-row / direct tile / direct tile |
| 128 | 63 / 64 / 65 | Per-row / per-row / per-row |
| 128 | 511 / 512 / 513 | Per-row / direct tile / direct tile |

Assert `GatherQmmTileF16` or `GatherQmmTileBF16` in each admitted cell and the selected per-row kernel in each fallback cell. Repeat across both storage dtypes, with repeated IDs and routing tails. The eight-expert cells isolate the minimum-64-row gate; the 128-expert 511/512/513 cells isolate the four-rows-per-expert gate.

For accepted offset views, use nonzero four-byte-aligned offsets for scales, biases and both index arrays. Verify the exact selected route and output bits. A two-byte scale or bias offset must return the existing compatibility error labeled `name() + " byte offset"` before direct or staged dispatch. Require the operation label, dtype and shape in that error. Record capability simulations separately from physical-device runs.

For the data-movement changes in #256 and #257, compare against their actual parent implementations, not a different arithmetic kernel. In #256, `MLX_OMARCHY_GATHER_QMM_DIRECT=0` also disables the tile. It therefore cannot isolate the old staged tile from the new direct tile. Use separate pinned builds for that comparison. Use `TILE=0` only as a separately labeled per-row correctness/performance control.

## 4. Correctness oracles and acceptance thresholds

### 4.1 Oracle capture

1. Capture native macOS MLX output on the same physical 14-inch M2 Max MacBook Pro for model throughput comparisons. Record chip, GPU core count, RAM, OS and native MLX identity. Another M1/M2 model cannot supply this denominator.
2. Store the exact input IDs, raw quantized operands, output tensors, logits, selected experts, cache state and recurrent state. Include shape, dtype, layout, offsets and byte digests. Select capture boundaries before looking at candidate errors.
3. Use independent fp64 contraction of the stored, dtype-rounded/dequantized operands for primitive checks. Keep this CPU oracle outside the release inference path. Comparing two Linux kernels alone cannot establish reference accuracy.
4. Capture baseline and candidate on the same teacher-forced token sequence. Validate the comparison instrument with an injected value/ID/state mismatch. It must identify the first divergent boundary and refuse acceptance.
5. Bind the oracle to a manifest containing exact numerical thresholds for each shape/dtype/state fixture. Missing oracle data, an absent threshold or a changed input is a failed qualification prerequisite, never a pass.

The [compiled/eager differential workflow](differential-harness.md) locates state and ordering defects. It does not replace the independent oracle when two implementations share the same numerical error.

### 4.2 Numerical gates

| Contract | Acceptance threshold |
| --- | --- |
| #256/#257 same-arithmetic data movement | Zero bit differences against the corresponding parent route for identical captured inputs, including output, logits and recurrent/cache state. Require the same routing sequence and raw generated IDs. No numerical tolerance or near-tie waiver applies to a new movement-induced difference. |
| Primitive oracle | Preserve every existing pinned bound. For gathered tile fixtures, existing relative-L2 limits are 0.01 for bf16 and 0.005 for fp16. These limits do not replace the zero-difference parent comparison above and must not be reused as blanket logit/state tolerances. Require finite values and report maximum absolute and relative-L2 error against the independent reference. |
| Dense QMM oracle | Retain the existing shape/dtype-specific host-reference bound in `test_matmul_family.cpp`. Do not relax it for the drain rewrite. Record actual storage rounding, K and worst error. |
| Arithmetic-changing candidates | Apply [numerics-gate.md](numerics-gate.md): per-op output/state error no worse than the deployed op against the same oracle; teacher-forced top-1 agreement at least baseline within the stated 95% binomial band; PPL deviation within baseline run-to-run noise. Freeze the sample count, interval method and noise calculation before comparison. Report counts, margins, PPL and mean absolute log-probability error. |
| Native token identity | Preserve every fixture already matching native. Apply [parity-id-policy.md](parity-id-policy.md) and its explicitly named amendments for other fixtures. Keep the legacy 32-token gate exact. |
| Safety | Zero NaN/Inf, out-of-bounds writes, CPU tensor dispatches, unexpected compatibility errors, GPU resets, device-loss errors or timeouts in required cells. A timeout is not a slow sample to discard. |

For arithmetic-changing candidates, the numerics gate treats a logit flip within two bf16 ULP of the top logit's magnitude as a near tie. Use the inclusive boundary and preserve the raw flip counts. Do not apply that exception to routing scores, state corruption, or the exact movement contracts. The existing one-ULP analyzer helpers require an explicit two-ULP comparator for this gate.

Use at least 5,120 predeclared teacher-forced positions per model for an arithmetic-change qualification. Report per-prompt results as well as the aggregate. For the existing 95% binomial-band rule, use the two-sided Wilson interval for baseline agreement and require candidate agreement at or above its lower bound. Count only the documented near-tie exceptions as additional accepted choices; retain raw exact-match counts separately.

Establish PPL noise from five identical-protocol baseline captures before examining candidate results. Define each deviation as `abs(log(PPL / PPL_reference))`; define the noise allowance as the maximum minus minimum baseline deviation. Require median candidate deviation no greater than median baseline deviation plus that fixed allowance. Record all five baseline values, candidate values and the calculation. These numerical gates do not waive exact native-matching or data-movement contracts.

## 5. Measurement manifest

Freeze the following before requesting the measurement window:

- Base/candidate source commits, upstream MLX pin, build options and wheel hashes; loaded `libmlx` hash, wheel RECORD agreement and distinct expected version stamps from [mlx_provenance.py](../scripts/mlx_provenance.py).
- Model and tokenizer manifest, mlx-lm source/patch hashes, prompt-ID files, teacher-forced corpus and expected output/correctness records.
- Physical device identity, kernel, firmware identity, private Honeykrisp ICD/library hash, driver/compiler versions and relevant environment overrides. Record shader cache policy, governor, power mode, clocks, thermal sensors and memory/swap state.
- Batch size, prompt lengths, output count, sampling, warmups, repetition schedule, phase timing definitions and deadlines. Record compilation, ANE, filler, model-serving and compositor states.
- Scheduler admission, lock ownership, start/end times, logging location and operator. Keep host addresses, serials and private service/topology details outside public records.

Verify that the two A/B arms are different builds and each matches its claimed commit. Different venv paths or filenames do not establish different binaries. Check provenance beside every result, not only during initial setup.

## 6. Five-run M2 window protocol

### 6.1 Admission and environment

A scheduler-issued measurement window must identify the physical M2 Max, allowed work, start/end times and recovery owner. A queued request, a plan or a previous window is not admission. The baseline owner maintains the native reference and coordinates the comparison record.

Before accelerator work, acquire the device admission lock and confirm competing GPU work is quiescent. Record the ANE/filler state and keep it identical across arms. Keep model servers off the measured device. Do not change the driver, governor, compilation policy, services, network or boot state between arms.

Use AC power and a fixed power policy. After warmup, settle for at least five minutes; begin paired arms within 2 degrees Celsius of the declared settled reference. Record sensor names and values. Declare any further load/thermal invalidation thresholds before execution. A thermal-policy violation invalidates the complete affected pair, with both raw runs retained.

Every process receives a finite deadline within the grant. Stop launching work when the remaining window cannot cover the full process deadline. On a hang/reset, stop the comparison, preserve logs and return recovery to the scheduler. Do not reboot or change hardware as an automatic benchmark recovery step. A later boot starts a new block.

### 6.2 Fixed cells and token inputs

Run both pinned models at batch size one, with 512 and 2,048 input tokens and exactly 128 output tokens. Report prefill and decode separately for every model/length. Do not average the small model with the MoE to hide a failed cell.

For each tokenizer, encode the fixed seed text `Explain how a computer stores and retrieves information. ` once. Repeat its resulting IDs and truncate to the requested input length. Freeze those arrays for both operating systems. Do not apply a second chat template. Validate vocabulary bounds and length. Compute SHA-256 over a little-endian uint32 count followed by little-endian uint32 IDs. Preserve complete arrays; decoded text cannot establish identity.

Use an explicit greedy sampler, temperature zero and seed zero. Disable EOS-based termination without changing the logit distribution. Assert 128 returned IDs, even if EOS appears. A shorter generation, empty prompt, invalid count or non-finite logit produces no accepted throughput value.

### 6.3 Process and timing rules

Each warm outer repetition is a new process with a freshly loaded and materialized model and fresh prompt/recurrent caches. Loading is excluded from warm throughput and recorded separately. Perform two untimed prompt-plus-eight-token warmups with fresh caches. Synchronize their completion. Capture cold metrics in the separate processes specified below.

Measure the following with monotonic host time and the same synchronization boundaries on both platforms:

1. **Prefill forward:** start after input construction and synchronization; evaluate the model's fixed prompt output and resulting cache/state before stopping. Use the same output/logit materialization policy on both sides. Rate = input-token count / elapsed time.
2. **First token:** on a separate fresh cache, start before the generation iterator's first evaluation; stop after the first token is materialized. This is warm time-to-first-token, not pure prefill.
3. **Decode:** record the completion time of token one and token 128. Rate = `127 / (t128 - t1)`. Record all 128 raw IDs. Include required host scheduling and token synchronization consistently; exclude loading and the first-token interval.

One timed forward and one timed generation per warm process supply that outer repetition's metrics. Optional internal repeats must be declared beforehand and retained individually; five internal forwards in one process are not five independent runs. Do not count warmups as repetitions.

For each arm and model/length cell, collect five additional fresh-process cold samples, separate from the five warm process pairs. Run no warmup in a cold process. Record monotonic timestamps before importing MLX (`t_start`), after runtime/device initialization (`t_load`), after model loading and full parameter materialization plus synchronization (`t_ready`), and after the first generated token materializes on a fresh cache (`t_first`).

Report initialization as `t_load - t_start`, cold model load as `t_ready - t_load`, cold time-to-first-token as `t_first - t_ready`, and total cold MLX start as `t_first - t_start`. Include file reads, dequantization/materialization and load-time compilation in model load. Include lazy forward compilation and pipeline creation in cold time-to-first-token. Record a fresh in-process pipeline/JIT cache and the declared driver disk-cache policy for every cold sample; keep that disk-cache policy identical across arms. Cold processes supply no warm throughput samples.

Keep dispatch tracing/profiling in separate correctness captures. Use the CPU-disabled qualification build from section 2.1 for Linux timing. Do not substitute the CPU-enabled oracle build, a diagnostic wheel or an instrumented driver.

### 6.4 Paired schedule and native comparison

For each model/length, predeclare five Linux pairs: `A1 B1`, `B2 A2`, `A3 B3`, `B4 A4`, `A5 B5`, where A is base and B is candidate. This supplies five independent processes per arm. Apply the same warmup/settle policy for each process. Never run both arms concurrently on the device.

A predeclared schedule may use more than five pairs. Do not extend a failing block until it crosses a threshold. If a new block is needed, record its purpose before capture and retain the failed block. Preserve every raw attempt and exclusion reason; replace a technically invalid pair as a whole.

Collect at least five independent native macOS processes per model/length on the same physical machine, with the identical manifest-controlled workload. The native capture can occur in a separate authorized window. Record per-OS conditions and elapsed time between captures. Historical results from another chip, quantization, prompt, output length or harness cannot fill an empty native cell.

Require matching raw free-run IDs before reporting a Linux/macOS free-run speed ratio. If numerical qualification permits a different stream, use a separate, explicitly labeled teacher-forced timing protocol on fixed native IDs. Never mix that rate with a free-run denominator.

### 6.5 Throughput acceptance

Apply these engineering decision thresholds to each required cell:

| Decision | Threshold |
| --- | --- |
| Valid cell | All correctness/provenance/safety gates pass; at least five valid independent processes per arm; within-arm spread `(max-min)/median` at most 5%. A noisier block is inconclusive. |
| Prefill candidate primary cells | Both 2B prefill lengths for #257, or both Nemotron prefill lengths for #256: median of paired `B_i/A_i` rates at least 1.03, with every valid pair above 1.00. |
| Regression controls | Median paired ratio at least 0.98 for every required non-primary model/phase/length cell; no correctness or safety regression. Declare the primary cells before execution. |
| Native throughput parity | Candidate Linux median / native macOS median at least 1.00 separately for decode and prefill in every required model/length cell. All workload and output-identity rules apply. |
| Compatibility release performance | Retain the separate pinned `llama.cpp` Vulkan comparison: `pp512` and `tg128` each at least 0.80, with its original workload and release repetition policy. This is not the macOS-parity criterion. |

Publish all rates, paired ratios, medians, spread and sample counts. Use the median of paired ratios for a candidate decision and the ratio of platform medians for a native comparison; label them distinctly. A geometric mean may summarize the complete matrix only after every mandatory cell passes. Never use it to waive a failed cell.

## 7. Test plan and acceptance records

| ID | Required scenario | Observable pass condition |
| --- | --- | --- |
| P1 | Source/build identity and same-head A/A control | Correct wheel/RECORD/source hashes; expected stamps. Five same-head pairs have median paired ratio in [0.98, 1.02] and each arm meets the 5% spread bound. No timer/control mismatch. |
| P2 | Independent oracle and detector negative controls | Captured operands and states resolve to the manifest; injected token/value/state mismatch is detected before an acceptance record is emitted. |
| P3 | Dense prefill drain matrix | All declared shader variants compile; CPU layout checks preserve one write per element; actual Apple cooperative-matrix output matches the parent bitwise across boundary/offset cases and repeated scratch reuse. Record actual kernel IDs. |
| P4 | Dense decode and recurrent transitions | Per-step logits, token IDs and recurrent/cache state meet the applicable exact/oracle contract across prefill, chunking, first decode, step 32 and step 128. Native-matching fixtures retain their IDs. |
| P5 | Sparse routing and expert boundaries | Top-six IDs/scores remain associated; Q4 packing/dequantization, sorted/shuffled routing, tails and both full expert dimensions pass captured-input checks. |
| P6 | MoE host dispatch, direct buffers and fallbacks | The section 3 matrix selects the expected kernel at every 8/128-expert boundary. MLX API dispatch preserves aligned parameter-offset/lifetime semantics and avoids parameter staging in direct cells. Misaligned scale/bias cases return the named error. Capability/flag fallbacks preserve reference accuracy. |
| P7 | Full hybrid model and no-CPU controls | Both models load pinned weights and produce validated outputs using the section 2.1 CPU-disabled wheel. Its GPU positive control and explicit CPU negative control pass in separate processes. Preserve the build-source and loaded-library records establishing CPU exclusion. |
| P8 | Five-run engineering matrix | All four model/length cells include five base and five candidate warm processes, separate phase metrics, separate cold samples, native records, raw IDs and valid admission/environment data. Apply section 6.5 per cell. |
| P9 | Integration and release regression | Recheck a clean integration commit against the same base. Run the standing chip regression gates before release. One hundred repeated requests cause no resets/timeouts/device loss and pass the retained-memory checks below. |

### 7.1 Stability memory measurements

Run stability separately from timing, once per model/length cell. Keep the model, compiled functions and fixed prompt alive. Use the same 128-token greedy request and cache policy for five warmup requests and the 100 measured requests.

1. Give each request a fresh prompt/recurrent cache. After completion, release its iterator, output tensors and cache references. Preserve output digests in the log rather than retaining tensors or request histories. Run Python garbage collection, synchronize all streams, then sample memory. Keep allocator-cache limits fixed and retain allocator/compiled caches throughout the run; do not clear caches or trim the host allocator between samples.
2. Sample post-warmup baseline `i=0` and every measured request `i=1..100` at that same boundary. Record `active_i = mx.get_active_memory()`, `cache_i = mx.get_cache_memory()` and `M_i = active_i + cache_i`, all in bytes. Record Linux resident memory `R_i` from `/proc/self/statm` resident pages multiplied by the recorded page size. Backend peak memory and process high-water RSS are separate diagnostic fields, not substitutes for these retained-memory series.
3. For each series `X` in `{M, R}`, require `X_100 - X_0 <= max(X_0 // 50, 16777216)` using integer division. This preserves the 2 percent or 16 MiB final-growth bound for both retained backend memory and process RSS.
4. For each series, compute the final-window integer numerator `S = 50 * sum(i * X_i) - sum(i) * sum(X_i)` over requests `i=51..100`. Require `S <= 0`. This is the sign of the ordinary least-squares slope; the allowed positive-trend tolerance is zero bytes per request. Preserve every sample, both numerators and both final-growth calculations so two readers obtain the same decision.

### 7.2 Acceptance records

An acceptance record contains the manifest digest, exact commands, start/end times, exit statuses, per-case results, raw output/state paths and byte digests. It also contains traces, kernel/driver diagnostics before and after, thermal/load logs, exclusions, fresh-process device reopen result, and the decision for each gate. Preserve failing and null results alongside successful results.

Physical-device qualification requires physical-device execution records for every required route. Update compatibility status only when the linked record covers every named gate. Keep each candidate's hardware acceptance record separate from the other candidates' records.
