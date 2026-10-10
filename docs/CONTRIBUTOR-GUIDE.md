# Contributor guide

This page is the answer to "what do you actually need help with?" Read it
and pick something. If after ten minutes you cannot find a task, this guide
failed. Open an issue and say so.

`mlx-omarchy` runs Apple MLX on Apple Silicon Linux through Mesa's
Honeykrisp Vulkan driver. The published v0.3.6 wheels support
`linux_aarch64` on M1 and `linux_x86_64` for development. The
[downloaded v0.3.6 wheel](../receipts/2026-09-06-release-v036/public-m1-generation.log)
measured 28.42 tok/s in one pinned 4-bit release smoke run.

The [development measurement](../receipts/2026-09-04-m1-performance-gates.md)
at `a34cdc5` reached 31.1 tok/s for short 4-bit Qwen2.5-0.5B decode.
[Native MLX 0.32.2 on the same M1](../receipts/native-baseline-2026-09-06/native-2026-09-06-nocompile-summary.json)
measured 150.8 tok/s with compilation disabled. These are distinct
measurements. They are not proof of functional or performance parity. See
the README for the workload table and the remaining token-identity mismatch.

If you have not read these yet, do that first:

- `README.md` (one-screen summary of what the project is).
- `docs/architecture.md` (what is in the backend).
- `docs/roadmap.md` (the proof-gated release plan).
- `docs/known-defects.md` (the live defect ledger).
- `AGENTS.md` (verification rules). For an outside contributor, the
  sections on "Test rules" and "Verification and receipts" are the ones to
  read.

## The hardware split, read this first

Hardware testing is authorized on the 13-inch M1 and 16-inch M1 Max Omarchy
laptops (`AGENTS.md` Hardware safety). Every Linux dev box without an Apple
GPU runs against `llvmpipe`/`lavapipe`. That is correct for structure,
dispatch counts, and code paths. It is meaningless for timing and
per-kernel numerics. This is not a wish. It is a hard split the rest of the
guide depends on.

| if you have | you can do | you cannot do |
|---|---|---|
| any Linux box (no Apple GPU) | shader-level reproductions, correctness tests, dispatch-count measurements, code reading, documentation, tooling, test additions, the `CONTRIBUTING.md` collector | real-Honeykrisp timing, hardware-only defect reports, shader micro-benchmarks on the Apple driver |
| M1 (or other Apple Silicon running Linux) | all of the above + hardware verification | (everything in scope) |

If you do not have an M1, the largest share of the open work below is
yours. If you do have one, you own the measurements. You own the regression
runs. You own the named-error triage that turn a code change into a
shippable result.

The shared M1 has one hardware owner at a time. Ask the session
coordinator for the current owner. Ask for an explicit access window before
you run commands. A deadline does not transfer ownership. The owner must
confirm that its processes have stopped. The owner must explicitly hand
back the machine. GPU access does not authorize ANE, driver, device-tree,
power management, or boot changes.

## What is unfinished

Open work, sorted by what it asks of you. Each item links to the receipt
that bounds it.

### Needs Apple hardware (M1), high value, harder to land

1. The 2,048-token queue wedge. It is live in v0.3.4
   (`docs/known-defects.md` "Live in v0.3.4"; see
   `receipts/2026-09-04-hang-watchdog-hardware.md`). One `mx.eval` over a
   full-sequence forward at 2,048 tokens wedges the GPU queue. The
   completion timeline counter stays frozen at 0. No work retires. The
   submission watchdog correctly names the hang. Chunked work at the same
   length is fine. `mlx_lm` is unaffected. What reaches it is a
   hand-written full-sequence forward. That is why no user has reported it.
   Hardware-only.
2. Subgroup arithmetic on the decode hot path. The kernel-level answer is
   no win. `tools/subgroup-bench` on the M1 clocked float `subgroupAdd`
   against the five-round shared-memory tree. The kernels differed only in
   the reduction body. Parity held at both tested sizes
   (`receipts/2026-09-04-qmm-gemv-subgroup-m1.md`). The reduction body is
   not a decode lever. The shipped 37% came from the GEMV dispatch. The
   variant stays staged at
   `receipts/2026-09-02-gemv-decode/matmul_vec_subgroup.comp`. It is
   correct only at real subgroup size 32. A device-subgroup-size gate is a
   precondition for any ship. Re-open only with a new mechanism, not a
   re-run.
3. The bf16 compiled-tape fusion fence. It is lifted with a pinned
   mechanism (`receipts/2026-09-18-bf16-compiled-tape.md`; see the
   `receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md`
   corrupt-matrix section). The 2026-09-02 tape-level bf16 refusal was the
   stale-shape defect. It is lifted as of 2026-09-18. The 2026-09-18
   fused-bf16 fence was the DivLast leaf misindexing. The fix landed at
   `da43969e`. Proof: the clean full-model mlx-lm sweep on the corrupt
   matrix with fusion ON. Qwen3.5 / Ministral / gemma-4-31B digests matched
   eager. Hardware-only.
4. The M1 verification bars for v0.4.0 and beyond. The v0.4.0 proof gate is
   the supported M1 matrix. v0.5.0 needs an M1 install of the public wheel.
   It needs MLX-LM generation with zero CPU dispatches. It needs one public
   workflow per ecosystem row. These are the bench runs that close out the
   next two releases. Hardware-only.
5. Dispatch census on real hardware, post-v0.3.4. v0.3.4 dropped 4-bit
   decode dispatches per token from 1,868.7 to 715.5 (-61.7%) via the
   fused-RoPE pair. Re-profile with `MLX_OMARCHY_GPU_PROFILE=...` on a
   clean main wheel. That is the env-gated harness. `scripts/profile_analyze.py`
   parses it. Then publish the new ranking. The fused chain shader at
   `shaders/fused_chain.comp` is in the tree. The question is whether it
   landed on the hot path. Hardware-only.

### Any Linux box, large surface, easier to land

6. Cut dispatch count. Use native-dtype elementwise paths. Use fused
   elementwise chains. Avoid materialized index arrays.
   `receipts/2026-09-02-gpu-profile-decode.md` shows Cast*, Copy*, and
   Arange at 35.8% of 4-bit dispatches. They also take 24-26% of GPU busy.
   Native bf16/f16 elementwise paths are the biggest single lever. They add
   no f32 round-trip. They carry zero ordering risk. Then build, run the
   llvmpipe battery, run `python3 scripts/bench_decode.py
   --self-test`, and write a value-test for the new paths.
7. Dependency-gated barriers. The blanket pre+post dispatch barrier pair
   costs ~35 us per intra-submission gap. Also, 71-75% of consecutive
   dispatch pairs are provably disjoint
   (`receipts/2026-09-02-gpu-profile-decode.md`). A dependency tracker is
   the second-order lever. It emits a barrier only when a dispatch's
   bindings overlap an unsynced written range. **A missed dependency is a
   silent wrong value.** It is the exact class that produced the v0.3.0
   defects. The tracker must be conservative. It must over-approximate.
   Gate it by name where uncertain. Add a dependency-trace test that
   asserts every refusal is reproduced after the swap.
8. Value tests for the four still-live defects in v0.3.1. `nn.gelu_approx`
   and `gelu_fast_approx` are pytest-context only. `fast_sdpa_vector`
   disagrees with its own decomposition. No standalone repro exists yet.
   One vjp path is off by one ulp. Each defect entry in
   `docs/known-defects.md` names the test file and line. Add a
   fresh-process repro for the first. Add a standalone probe for the
   second. Add a tolerance note for the third.
9. Documentation of the verification rules for outside readers.
   `AGENTS.md` is written for agents. A short "How to verify what you just
   changed" page is missing. It would aim at a person who has never run
   this stack. It would shorten every new contributor's ramp.
10. CI. The daily community-data mirror is automated. Test and release
    gates run locally. A self-hosted runner would be a small, high-value
    infrastructure piece. It would build the dev-box wheel. It would run
    the full llvmpipe battery. It would run `bench_decode.py --self-test`.
    Speak to Main before touching `.github/workflows/`.

### Real work for an M1-equipped contributor who has 30 minutes

- Re-run `scripts/bench_decode.py --wheel <wheel>` against current main on
  your M1. Print provenance beside the result. Record the numbers in
  `receipts/` with the exact source, wheel, and protocol.
- Run `python3 scripts/differential_compile.py --self-test` against the
  same wheel. Copy the output.
- Pick one defect from `docs/known-defects.md` "Live in v0.3.1". Find the
  smallest reproducer. Add a C++ unit test that fails today and passes
  after the fix lands.

## What has already been tried (and did not ship)

These were tried, measured, and stood down. A contributor who re-proposes
one of these wastes their time and ours. The numbers are pinned. The reason
it lost is in the linked receipt.

| idea | result | why it lost | receipt |
|---|---|---|---|
| Dispatcher-wakeup polling (1 ms in-flight poll) | bf16 decode -2.5%, 4-bit decode +0.3% on M1 | within noise; the hypothesis was wrong | `receipts/2026-09-03-dispatcher-compile-and-column-replace.md` |
| Generic tape fusion at a 7.3% ceiling | fused-fragment path covered only 7.3% of model shape changes; rest refused by name | too narrow to be worth the extra compiler surface | `receipts/2026-09-03-dispatcher-compile-and-column-replace.md` (Compiled path: two gates, two generations) |
| Compile-the-forward alone (`ff4b05a` ordering wait, no other changes) | bf16 decode -27.5%, 4-bit decode -39.2% vs parent `4ea2f47` | the ordering fix is not free on hardware; the regression is real and ships with the correctness benefit | `receipts/2026-09-03-dispatcher-compile-and-column-replace.md` |
| Host scalar folding (`fast::rope` host-side offset probe) | bit-exact at the offset position; the earlier divergence was the probe, not the primitive | refuted by the corrected probe; rope stands | `docs/known-defects.md` "The rope divergence: the probe, not the primitive" |
| Deep batching (16-100 ops per submission) | 4.5x slower on llvmpipe (0.637 vs 2.838 prompt tok/s); slower on M1 | per-op flush makes the throttle wait span one op; batched flush makes it span the whole batch; code deleted from the diff | `receipts/2026-09-03-submission-ring-descriptor-cache.md` "Batching: implemented, measured, DELETED" |
| Software Payne-Hanek for `mx.sin`/`mx.cos` above 1e5 | on M1 returns -7.9e15 for sin(5e6) | the carry chain rides the same dynamic-indexing miscompile class as the masked-scatter defect | `docs/known-defects.md` "mx.sin and mx.cos degrade above 1e5..." |
| ~~GEMV decode path~~ - the kill was wrong | stood down 2026-09-02 on timings taken with 1 of 8 CPU cores online; re-measured 2026-09-04 with all cores: 4-bit decode 12.83 -> 17.61 tok/s (+37%), bf16 unchanged | shipped on main as `qmm_vec.comp`. Listed here so nobody trusts the 2026-09-02 verdict: a strategy killed under a contaminated condition is not killed | `receipts/2026-09-04-qmm-gemv-subgroup-m1.md` |

A variation of one of these is not a fresh angle. It inherits the loss. New
ideas, please.

## Open questions (what would answer each)

These are the live questions the project needs answered to make the next
release.

1. Why does a single 2,048-token `mx.eval` wedge the GPU queue with the
   timeline frozen at 0? The watchdog correctly classifies it as a hang.
   The wedge reproduces on the M1. Chunked work of the same length is
   unaffected. `mlx_lm` is unaffected. The defect entry is
   `docs/known-defects.md` "Live in v0.3.4". A reproducer would answer it
   if it names the resource the queue waits on. Candidates: descriptor,
   semaphore, fence, allocator page, in-flight submission. Likely inputs:
   shape threshold, allocator pool exhaustion, an in-flight submission that
   the new submission stalls behind.
2. Why are our kernels roughly six times slower than Metal's on the same
   silicon? GPU busy per 4-bit decode token is 19.5 ms on our tree. Metal
   totals 3.4 ms (q4 decode profile;
   `receipts/2026-09-02-gpu-profile-decode.md`). The four-line candidates
   follow. Shared-memory barrier tree (`barrier()` vs `subgroupAdd`).
   Unroll policy: Honeykrisp is register-bound, and `llama.cpp` disables
   unrolling on this driver. Register-tiling width. `gl_WorkGroupSize`
   declaration: this driver rejects implicit reads, which is what broke the
   v0.3.4 fused-RoPE shader on `glslc`. A targeted kernel-vs-kernel A/B on
   the M1 would decompose the 6x. Isolate each lever.
3. Does subgroup arithmetic close the kernel gap? The kernel-level answer
   is no. Float `subgroupAdd` and the shared-memory tree cost the same on
   the M1 (`tools/subgroup-bench`; see
   `receipts/2026-09-04-qmm-gemv-subgroup-m1.md`). The 6x decomposition
   (question 2) is the live work.
4. What are the remaining ~4 host copies per layer per token, and who owns
   them? The GPU profile shows dispatch record cost is now 1.7 us median on
   llvmpipe. It was 160-260 us per dispatch
   (`receipts/2026-09-03-submission-ring-descriptor-cache.md`). The
   host-side residual on the M1 is the next ceiling. The profiled hot path
   lives in `overlay/mlx/backend/omarchy/encoder.cpp` (Begin / commit /
   descriptor pool cache). A per-encoder trace with
   `MLX_OMARCHY_GPU_PROFILE=...` + `scripts/profile_analyze.py` names which
   step is the ceiling.

## Verification bar (briefly)

"Measured on hardware or it does not ship." That sentence is not
bureaucracy. In one 24-hour period, six measurements from this project
described an environment nobody intended. A stale wheel reinstalled by a
freeze pin. A polluted build dir. A single-core run masquerading as the
supported configuration. A test reference computed into `std::vector<bool>`
(bit-packed, miscompiled by `g++ 16.1.1` aarch64). A shader that compiled
green on the dev box (`glslangValidator 12.0.0`) and failed on the M1
(`glslc 2026.3`). Each looked entirely plausible. The provenance line
beside every number (`scripts/mlx_provenance.py`) exists because of this.
So does the per-wheel A/B discipline: wheels stamp their source commit. So
does the rule "rebuild after every rebase because `.work/mlx` holds a
copy". They exist because they were the bugs. Read `AGENTS.md` sections
"Test rules" and "Verification and receipts" before you push. A small test
that proves a real defect is worth more than a paragraph that explains one.

## Entry points: concrete commands

These are the commands a contributor runs first. Each was verified on the
dev box at the time this guide was written. If a command fails, file an
issue with the exact error and the wheel commit.

### Pull the source and prepare the pinned MLX

```sh
git clone https://github.com/joshuaswarren/omarchy-mlx.git
cd mlx-omarchy
./scripts/prepare-mlx.sh        # fetches mlx.lock's archive, sha-checks it, stages .work/mlx
```

`prepare-mlx.sh` is the only sanctioned way to inspect the pinned upstream
tree. Do not vendor `mlx` source into this repo.

### Build the wheel

```sh
# Release wheel (profiling harness compiled OUT):
DEV_RELEASE=1 ./scripts/build-wheel.sh

# Diagnostics wheel (profiling harness compiled IN, slower, profiling only):
./scripts/build-wheel.sh --diagnostics
```

The release wheel stamps its source commit into the version's local segment
(`0.32.2.dev<timestamp>+<short7>`). The diagnostics wheel stages
`+diag.<short7>`. Verify both.

### Self-test the measurement scripts before you trust them

```sh
python3 scripts/mlx_provenance.py --self-test
python3 scripts/bench_decode.py --self-test
python3 scripts/differential_compile.py --self-test
```

All three exit 0 on the dev box today (verified 2026-09-04).

### Run the dev-box battery against llvmpipe (no Apple GPU needed)

The full battery is 24 binaries. See `AGENTS.md` "Test rules" for the build
flags. Run from `.work/mlx`:

```sh
cd .work/mlx && ctest --output-on-failure -j$(nproc)
```

### Run the dispatch census (no Apple GPU needed)

Counts and groupings work on llvmpipe. Timing does not. Install the
diagnostics wheel in a venv first:

```sh
export MLX_OMARCHY_GPU_PROFILE=$HOME/mlx-profile.jsonl
~/venv-mlx-diag/bin/python -m mlx_lm generate \
  --model /path/to/Qwen2.5-0.5B-Instruct-4bit-mlx \
  --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 32 --temp 0 --seed 0

python3 scripts/profile_analyze.py \
  $HOME/mlx-profile.jsonl \
  --compute-h overlay/mlx/backend/omarchy/compute.h
```

On a non-Apple box the path is the same. The numbers are correct for
structure and dispatch count only.

### Publish a community hardware report (no write access needed)

```sh
python3 scripts/collect_quick.py
# or with --submit <URL> to publish to the community dataset
```

Full path lives in `CONTRIBUTING.md`.

### Hardware protocol for an M1-equipped contributor

```sh
# 1. Confirm core count before you bench
nproc && cat /sys/devices/system/cpu/present

# 2. Install the wheel into a venv
python3.14 -m venv ~/.venvs/mlx-collect
~/.venvs/mlx-collect/bin/pip install \
  https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.3.5/mlx_omarchy-0.32.2.dev202609040917%2B0535e62-cp314-cp314-linux_aarch64.whl

# 3. Print the provenance line beside every measurement
~/.venvs/mlx-collect/bin/python scripts/mlx_provenance.py

# 4. Run pinned decode (36-token prompt, 64 pinned tokens; take the median of five invocations)
~/.venvs/mlx-collect/bin/python scripts/bench_decode.py \
  --model /path/to/Qwen2.5-0.5B-Instruct-bf16-mlx \
  --prompt "What is the capital of France? Answer in one word." \
  --tokens 64 --temp 0 --seed 0 \
  --wheel <wheel-file-you-just-installed>
```

## Where this lives

This page is `docs/CONTRIBUTOR-GUIDE.md`. `CONTRIBUTING.md` keeps its
current focus. That focus is community data collection, source rules, and
proof required. The README links to this page from the documentation index.
The two documents overlap on the verification bar. They deliberately do not
duplicate it. `AGENTS.md` is canonical for the exact rules. This page is
the contributor-facing summary with the why.
