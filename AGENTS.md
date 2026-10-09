# Agent instructions

## Start here

Read these files before changing code:

1. `README.md`
2. `docs/architecture.md`
3. `docs/roadmap.md`
4. `docs/compatibility.md`
5. `docs/plans/2026-08-29-mlx-omarchy-ane-compatibility-plan.md`
6. `docs/plans/2026-09-12-coreml-parakeet-ane-plan.md` (Core ML/Parakeet
   plan; package inspection, reference capture, and compiler preparation
   exist in source. Linux encoder execution remains unqualified.)

Run `scripts/prepare-mlx.sh` before inspecting the pinned MLX source.
Inspect the affected backend under `.work/mlx`.
Keep project-owned files under `overlay/` and upstream edits under `patches/`.

Use the CUDA backend for the complete GPU boundary.
Use `mlx/backend/no_cpu/` and `mlx/backend/no_gpu/` for failure behavior.

## Fixed product contract

- Keep the public MLX Python import as `mlx`.
- Keep `mx.gpu` as the public accelerator device.
- Use Honeykrisp Vulkan for complete tensor execution.
- Treat ANE as an internal graph-region accelerator.
- Let CPU code schedule, copy, tokenize, and serve requests.
- Never run a tensor primitive on CPU in a release build.
- Return an exact compatibility error when Vulkan and ANE cannot run a primitive.
- Do not emulate Metal on Linux.
- Qualify M1 Omarchy before other Apple Silicon models.

Do not weaken these rules to make a model run.
Add the missing accelerator implementation or keep the test failing.

## Repository boundaries

- This repository owns the Omarchy backend, ANE partitioner, patches, packaging, tests, and releases.
- Upstream MLX source and history must not enter this repository.
- `joshuaswarren/ane-linux-experiments` owns hardware probes, format research, and unstable fixtures.
- `joshuaswarren/omarchy-ane` owns the ANE DRM driver and `libane` ABI
  (upstream lineage and backport flow: `docs/forks.md`).

Do not copy experimental driver code here.
Prove a driver change in the experiment repository, then send the smallest stable ABI change to `joshuaswarren/omarchy-ane`.

## Upstream MLX sync

Pin the upstream release, commit, archive URL, and SHA-256 in `mlx.lock`.
Fetch upstream source only through `scripts/prepare-mlx.sh`.
Add new project files under `overlay/`.
Keep modifications to existing MLX files in a small patch series.
Rerun runtime, primitive, and semantic gates after every baseline change.
Keep Omarchy work behind `MLX_BUILD_OMARCHY`.
Do not rename public MLX APIs to fit backend code.
Resolve an upstream change at the shared MLX contract before adding an Omarchy-only path.

This repository is independent from Apple.
Do not open an upstream MLX pull request or push to an Apple remote from an agent session.

## Implementation order

Work in the dependency order in `docs/roadmap.md`.
The Vulkan baseline does not wait for ANE.
Owner direction, 2026-09-12: the Core ML/Parakeet ANE lane
(`docs/plans/2026-09-12-coreml-parakeet-ane-plan.md`) runs concurrently
with GPU performance-parity work; neither lane waits on the other's
numbers. The correctness gates do not move: the ANE exporter still
waits for the MIL compiler proof, general MLX lowering still waits for
the one-operation MIL proof and the stable `libane` ABI, and the
exporter still does not wait for dma-buf support.

Implement primitives in model-driven order, but keep each primitive general.
Do not add a model-name branch inside a core tensor kernel.
Reuse backend-neutral MLX code before adding Omarchy code.
Run the U2 matmul and attention speed spike before broad primitive work.
Stop and revise the backend design if the spike misses its fixed go or no-go gate.
Partition ANE regions before `mx.compile` fusion.
Represent each selected region as an opaque `AneRegion` primitive.
Send the rest of the graph through Vulkan fusion.

## Hardware safety

The owner authorized both the 13-inch M1 and 16-inch M1 Max Omarchy
laptops for installation, hardware testing, and necessary reboots
(2026-09-12). Consult the private fleet inventory for SSH targets.
Use an independent `/tmp/m1-gpu.lock` on each physical host. Exclude a
serving host from LiteLLM and confirm it is quiescent before benchmarking.
Native performance comparisons must match the physical machine; never
use the base M1 as the M1 Max denominator.

**Hardware status (2026-09-20).** m1-test-host (13-inch M1) runs provisioned
Omarchy; its Linux ANE qualification ladder has Step 5 closed — full-ASR
104/104 across two bit-exact repeats on the fork driver and a 10/10
perf battery (encoder AC median 5217.4 ms vs 259.9 ms macOS same-encoder
divisor, cross-OS indicative, no parity claim;
[`ane-linux-experiments` 3babdb5](https://github.com/joshuaswarren/ane-linux-experiments/commit/3babdb5)
/
[119b954](https://github.com/joshuaswarren/ane-linux-experiments/commit/119b954)).
Hybrid partition and full-encoder coverage are pending. Pre-provisioning
M1 ANE / Parakeet numbers in this tree are historical dated evidence from
prior m1-test-host Linux boots. t6001-test-host (T6001) Linux ANE is live. t6021-test-host (T6021)
Linux reads kernel 7.1.13-3-1-ARCH stable, ANE_UNBOUND, no
`/dev/accel/accel0`. T6021 GPU is qualified
([`ane-linux-experiments/receipts/2026-09-23-m2-gpu-qwen38/README.md`](https://github.com/joshuaswarren/ane-linux-experiments/blob/main/receipts/2026-09-23-m2-gpu-qwen38/README.md));
T6021 ANE is **not** live-inference-qualified on the Linux driver
(macOS-side numbers are macOS CoreML / `aned` measurements, not Linux
execution). Apple GPU and Apple ANE are separate lanes.

New descriptor, synchronization, and dma-buf tests can reset the device or machine.
Use a bounded timeout for every new hardware path.
Run one new failure mode at a time.
Record the device state before and after the test.
Run ANE submits in a bounded worker that owns the device file descriptor and resident buffer objects.

Do not enable dma-buf mode by default until export, import, coherency, fencing, and recovery tests pass.
Use shared host staging buffers between the evaluator and worker.
Include staging copies and process IPC in every ANE crossover result.
Do not redistribute Apple private frameworks, compiler binaries, firmware, credentials, or model weights.

## Verification and receipts

A code claim needs command output from the same session.
A hardware claim also needs a receipt file.
Each hardware receipt must include:

- source commit
- kernel and Mesa versions
- Vulkan device and firmware identity
- ANE driver and compiler identity when applicable
- model and quantization hashes
- exact command
- numerical result
- backend dispatch trace
- timing and thermal procedure for performance claims
- Vulkan device reopen result or ANE worker, reboot, bring-up, and self-test recovery result

Generate `docs/compatibility.md` status from receipts when tooling exists.
Until then, update a row only when the linked receipt proves every named gate.


## Public writing

Published docs, READMEs, model cards, release notes, and posts state what works and what it requires.

- Do not narrate process or hedge: no "pending proof" or "pending replication", no "receipt, not proof", no "not yet verified on X", no "I didn't test", no provenance caveats, and no "evidence" framing.
- If something does not work, fix it or leave it out. Do not publish a caveat about it.
- State requirements plainly: RAM, OS, kernel, and hardware (for example, "96 GB of unified memory; 64 GB machines can't run it").
- Do not mention Asahi in public prose. Technical identifiers (paths, device-tree properties, upstream repo names) stay as they are.
- Receipts and internal notes under `receipts/` and `receipts-work/` keep their full detail; this rule covers what readers see first.

## Test rules

- Print the provenance line beside every measurement, reproduction, or
  defect claim, including throwaway probes. `scripts/mlx_provenance.py`
  hashes the loaded `libmlx.so`, compares it against the installed
  wheel's RECORD, and reports the harness commit. A venv can silently
  hold a wheel from another tree: on 2026-09-03 that produced four
  separate wrong conclusions in one day - a retracted byte-identity
  claim, a phantom missing profiler, a stale-checkout crash that idled
  the M1 for 47 minutes, and a reported eager-evaluation defect that did
  not exist. Each looked entirely plausible. An unlabelled number is not
  evidence, and a defect report from an unverified environment costs
  more than no report.
- An A/B run must prove its two sides are different builds before it
  reports numbers. Wheels stamp their source commit into the version
  (`0.32.2.dev<timestamp>+<short7>`, diagnostics `+diag.<short7>`), so
  assert that each side's stamp equals the commit that side was meant to
  be, and that the two stamps differ. Fail with both stamps printed. On
  2026-09-03 two comparisons were voided after the fact - one built the
  wrong branch because a helper resolved HEAD instead of the branch
  name, the other built both sides from the same tree - and a third run
  died on a shared cmake cache left behind by an earlier build. Clear
  the cache per build for the same reason: an environment that lies
  about what it built produces numbers that look fine.
- Add one focused test for each new observable contract.
- Run the release-equivalent build with CPU primitive evaluation unavailable.
- Trace every ecosystem workflow at least once and require zero CPU tensor dispatches.
- Compare numerical output with a pinned macOS MLX build or locked fixture.
- Use the plan's exact Qwen reference model for the 32-token contract.
- Use pinned numerical tolerances and fixture-specific argmax or top-k checks for other models.
- Compare performance only on the same machine, model, quantization, prompts, and thermal procedure.
- Keep cold-start, warm steady-state, prefill, decode, and first-token results separate.
- Trace recurrent ANE state reuse and residency invalidation across decode steps.
- A shader that compiles on the development box may fail on the M1,
  because the two machines run different shader compilers. The build
  prefers `glslc` and falls back to `glslangValidator`
  (`overlay/mlx/backend/omarchy/CMakeLists.txt:10-27`). The x86
  development box has `glslangValidator` 12.0.0 and a `glslc` shim
  (`/usr/local/bin/glslc`: a 10-line script over a glslang 16.6.0 build in
  `/opt/glslang-src`; it is NOT shaderc's glslc, but it is much closer to the
  M1/M2 `glslc` 2026.3). Use the shim for every compile-only check of a new
  shader (it compiles cooperative-matrix shaders but cannot run them).
  m1-test-host has `glslc` 2026.3. The older 12.0.0 validator accepts
  constructs the newer compiler
  rejects. On 2026-09-03 a fused RoPE shader read `gl_WorkGroupSize`
  with no `local_size` declaration, compiled clean here behind three
  green batteries, and broke the aarch64 wheel build on the M1.
  `add_dependencies(mlx omarchy_shaders)` means the batteries DO compile
  shaders, so a green battery is real evidence about this box's compiler
  and no evidence at all about the M1's. Before pushing a change under
  `shaders/`, either build it where `glslc` is present or have the M1
  owner build it.
- Re-run `scripts/prepare-mlx.sh` after any rebase before building. The
  prepared tree under `.work/mlx` holds a copy of the overlay, so a
  build after a rebase without it tests the code you had, not the code
  you have.
- M1 qualification windows (driver, full-coverage, decode-prefill) must
  run the standing M1 battery, not an ad-hoc subset. The standing M1
  battery is the suites under `overlay/tests/omarchy/` that exercise
  every omarchy-backend route: `omarchy_runtime_tests`,
  `omarchy_primitive_tests`, `omarchy_matmul_family_tests`,
  `omarchy_fast_ops_tests`, `omarchy_kv_ops_tests`,
  `omarchy_indexing_ops_tests`, `omarchy_reduce_ops_tests`,
  `omarchy_shape_ops_tests`, `omarchy_linalg_ops_tests`,
  `omarchy_copy_offset_tests`, `omarchy_distributed_tests`,
  `omarchy_compiled_tape_tests`, `omarchy_fft_ops_tests`,
  `omarchy_fft_general_tests`, `omarchy_eig_ops_tests`,
  `omarchy_take_fill_tests`, `omarchy_conv_tests`,
  `omarchy_complex_ops_tests`, `omarchy_select_layout_tests`,
  `omarchy_fast_regression_tests`, `omarchy_scatter_determinism_tests`,
  `omarchy_eq_math_tests`, `omarchy_fused_chain_tests`,
  `omarchy_error_contract_tests`, `omarchy_ane_bundle_tests`, plus the
  `omarchy_capability_sim_tests` profile matrix. The prior pattern of
  running only `omarchy_matmul_family_tests` +
  `omarchy_runtime_tests` left the f16 SDPA route untested on the M1 for
  months and let a simulation-only refusal pattern be mistaken for a real
  defect (`receipts/2026-09-11-f16-sdpa-gap-analysis`).
- This repository is public. A receipt carries the evidence a reader needs
  to judge a measurement, and nothing about the owner's private
  infrastructure. Never commit host addresses (LAN or VPN), service
  inventories, ports, backup targets, keychain or credential observations,
  hardware serials, or the names and topology of machines and model
  endpoints that serve anything outside this project. Record the chip, OS,
  driver build, and a placeholder for the host; `scripts/collect_common.py`
  has a `Redactor` that shows the intended shape. Harnesses that sample
  `ps`, `system_profiler`, or `tmutil` must record counts and verdicts, not
  raw output: on 2026-09-11 the native-baseline harness had put process
  command lines, a NAS backup URL and a battery serial into published
  receipts, and a planning doc named the owner's router container, config
  path and model aliases (scrubbed forward in
  `receipts/2026-09-11-public-repo-scrub`).

Do not lower tolerances, shorten a stability run, or remove a failing workload to make a gate pass.

## Documentation

Update `docs/architecture.md` when a backend or memory boundary changes.
Update `docs/roadmap.md` only when a proof gate changes.
Update `docs/compatibility.md` from evidence, not intent.
Delete stale paths and replaced design text in the same change.

## Community hardware data

Contributors submit redacted hardware reports to the public dataset.
Submissions never create pull requests, issues, or discussions, and the
mirror commits snapshots as `github-actions[bot]`, so the contributor
graph stays clean. Query other people's machines before you guess at
hardware behavior.

Query with the stdlib CLI (no install, no network needed for local
mode):

```bash
python3 scripts/query_community_data.py list
python3 scripts/query_community_data.py --json --kind deep --chip M1 list
python3 scripts/query_community_data.py --kernel 7.1.6 --json list
python3 scripts/query_community_data.py --mesa honeykrisp list
python3 scripts/query_community_data.py --mlx-version 0.3.2 list
python3 scripts/query_community_data.py show <sha256-prefix>
python3 scripts/query_community_data.py compare --metric tflops
python3 scripts/query_community_data.py compare --metric median_ms --size 512
```

`--json` prints machine-readable output for agents. Filters take
case-insensitive substrings and combine. Sources:

- `--source auto` (default): local snapshot when present, else the live
  public endpoint.
- `--source local`: the mirrored snapshot only. The ladder is
  `--snapshot DIR`, then `$MLX_OMARCHY_DATA_DIR`, then
  `community-data/snapshot/latest.jsonl` in this checkout, then the
  fetched `origin/community-data` ref read through `git show`, so
  `git fetch origin community-data` is enough; no worktree needed.
- `--source remote`: the live public endpoints, base URL from the
  repository variable `COMMUNITY_DATA_BASE_URL`, default
  `https://mlx-omarchy-community-data.joshua-s-warren.workers.dev`:
  - `GET /v1/results` - index: generated_at, schema_version, count
  - `GET /v1/results/<sha256>` - one full record
  - `GET /v1/results/<sha256>/archive` - original redacted archive
  - `GET /v1/dataset/latest.jsonl` - one JSON object per line

The scheduled `.github/workflows/community-data.yml` mirrors summaries
(never archive blobs) to the `community-data` branch under `snapshot/`:
`latest.jsonl`, `index.json`, and a generated `SUMMARY.md`.

Treat the data for what it is: redacted, self-reported, one machine per
record. Compare performance only within the same model, quantization,
and measurement procedure, and name the records a claim rests on.
