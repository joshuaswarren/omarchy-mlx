# In-place 4-bit and 8-bit field decode in `qmm_vec.comp` (M4 Mac mini)

Result: the decode GEMV's magic-number nibble decode no longer shifts each field down and ors it into the exponent of 2^23. The field is masked in place and read as the mantissa of the float whose unit sits at its bit position, so the same float comes out of an `and`, an integer `add` and a subtract. Every product and sum is unchanged: `qmv_bits.py` gives 120/120 decode `quantized_matmul` cases bit-identical to `main`, and the generated token ids are identical on every model below. On the M4 Mac mini the 4-bit decode GEMV runs 3 to 6 percent faster at the larger shapes (2048 x 2048: 73.2 -> 69.2 us; 8192 x 2048: 237.3 -> 222.5 us) and decode gains 3.4 percent on granite-4.0-1b-4bit and 6.4 percent on Qwen2.5-1.5B-Instruct-4bit; the 8-bit model is unchanged within its spread.

## What changed

`shaders/qmm_vec.comp` only: `Q4_NIB`, `Q8_BYTE` and `Q4_NIB_WORD`.

- A field n at bit s, masked in place, is n * 2^s. Read as the mantissa of the float with unit 1 at bit s, that is 2^(23 - s) (exponent word `(150 - s) << 23`), it is 2^(23 - s) + n, so subtracting 2^(23 - s) returns n exactly.
- The exponent word goes in with an integer add; the bits are disjoint, so add equals or.
- Valid while the field sits below bit 20: every `Q4_NIB` use is at shift 0 to 12, and `Q8_BYTE` shifts bytes at bit 16 or 24 down by 16 first, as before. `Q4_NIB_WORD` takes the high half word through the same `>> 16`.
- On G16G the GEMV loop goes from 32 right shifts to 7 and from 39 ors to none. `op_rate.py` (in this directory; see Speed) gives the issue rates that make that matter here.

No host code, route, test or flag changes. There is no new observable contract, so no new test: the existing `quantized_matmul` value tests and `test_qmv_batch.cpp` (batch rows against the single-row kernel, bit for bit) cover the kernel.

## Host and stack (`host.txt`)

- Apple Mac mini (M4, 2024), `apple,j773g` / `apple,t8132`, 16 GB. Omarchy, kernel `7.1.13-1-20-ARCH` with the `asahi_m4` G16G GPU driver (GPU ID `0x0a021100` revision `0x11`, boot firmware `mBoot-18000.161.10`). PMP not bound.
- Vulkan: Mesa 26.3.0-devel Honeykrisp (`git-ea328336ef`, techmore/mesa `m4-honeykrisp-compute` at `ea32833`), `Apple M4 (G16G B1)`, API 1.4.359. `AGX_APPLE9_MAX_SHARED=32768`, `MLX_OMARCHY_BATCH_WORK=120000`.
- shaderc `glslc` 2026.4 on the host; all 39 `qmm_vec.comp` targets compiled in the wheel build.
- Python 3.14.8, mlx-lm 0.31.3.

## Builds (`wheels.txt`)

- A: upstream `main` `a69d9e6` (`0.32.4.dev202611020134+a69d9e6`).
- B: this change, code commit `ef78016` (`0.32.4.dev202611020155+ef78016`).
- Fresh worktree, `scripts/prepare-mlx.sh`, new CMake tree per build (`scripts/build.sh`). Every process printed the provenance line and refused unless its stamp matched the intended commit (`scripts/prov.py`, `scripts/ab.py`). `scripts/ab.py` stops on the first process that exits non-zero (provenance refusal 3, timeout 124, any other status) or adds a GPU driver error: it writes that record, runs no later arm and exits with the child's status. GPU error monitoring is required: if `journalctl -k` cannot run or exits non-zero, `ab.py` refuses with status 8 before any arm runs and writes no record (after an arm, it writes that record with the monitor error and stops with 8). `scripts/test_ab.py` checks the rc 3, rc 124, absent-monitor and failing-monitor cases without a GPU. The records in this receipt were written by the earlier runner, which logged such a status but kept going; every record here has `rc` 0 and no new kernel error. The monitor was live on the host: `scripts/facts.sh`, run as the same user, read the GPU driver's kernel lines through the same `journalctl -k` (`host.txt`).

## Tests (`battery.txt`)

This change (`ef78016`, the head's code commit): `omarchy_matmul_family_tests` 42/43 and `omarchy_qmv_batch_tests` 4/4. The one failing case (`gather_qmm sorted-expert tile route matches the f32 reference and the per-row route`) fails the same way on `main` on this host (an evaluation error in the gather tile route, not in `qmm_vec`).

The suites above are not an all-green battery: `omarchy_matmul_family_tests` is 42/43 here, with the same named case failing on `main`, and `omarchy_qmv_batch_tests` is 4/4. On this host `main` fails 34 of 762 cases and the six-change integration the same 34 of 800 (`receipts/2026-11-01-m4-g16g-part1/battery.txt` names them). `omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which upstream marks `doctest::may_fail` (an expected failure); its passing case total does not make it a clean suite. The dispatch traces below come from profiling builds of `main` and of the six-change integration, so they show which kernels ran and their per-dispatch cost, not this PR's isolated share of decode time.

## Exactness (`bits-p3-vs-main.log`)

`qmv_bits.py dump` on A, `compare` on B: 4-bit group 64 and 32 and 8-bit group 64 and 32; bf16 and f16; 1, 2 and 4 rows; five shapes including K not a multiple of 512 and N not a multiple of 8. 120/120 bit-identical. The one-row cases run the changed `qmm_vec` kernels and the 2- and 4-row cases the cooperative-matrix prefill tile; `engage-qmv-kernels.txt` (a profiling run of the same cases on the integration) lists `QmmVecQ4WordSubgroup{BF16,F16}` and `QmmVecWordSubgroup{BF16,F16}{Q4G32,Q8G64,Q8G32}` among the dispatched kernels. `compare` exits 1 on any differing or missing case (`bits-selftest.log`: 0 against a matching baseline, 1 against one with a flipped value). The RoPE (72/72), decode SDPA (35/35) and bf16 Q4 prefill (21/21) compares in the same log are identical too, with device reopens before and after.

## Speed

Decode (`perf-p3-summary.txt`): `scripts/decode_bench.py <repo> 64 4`, three interleaved A/B pairs (A B, B A, A B), median of process medians; idle and temperature procedure as recorded per process (load below 0.5, 20 s idle, NAND and Wi-Fi SMC sensors 32 to 42 C).

| model | A main | B this change | B/A | token ids |
|---|---:|---:|---:|---|
| granite-4.0-1b-4bit | 38.82 | 40.13 | 1.034 | equal |
| Qwen2.5-1.5B-Instruct-4bit | 44.78 | 47.67 | 1.064 | equal |
| Llama-3.2-1B-Instruct-8bit | 28.11 | 28.02 | 0.997 | equal |

The 8-bit model's B processes read 27.99, 29.34 and 28.02, so its change is inside the spread.

GEMV (`perf-p3-gemv.jsonl`): `gemv_sweep.py 7`, production `quantized_matmul`, 4-bit group 64 bf16, many independent copies in one graph, median of 7; two interleaved pairs, per-call microseconds (mean of the two processes):

| N x K | A main | B this change |
|---|---:|---:|
| 512 x 2048 | 33.6 | 32.3 |
| 512 x 8192 | 50.4 | 49.0 |
| 2048 x 512 | 62.8 | 60.1 |
| 2048 x 2048 | 73.2 | 69.2 |
| 2048 x 8192 | 141.9 | 138.2 |
| 8192 x 2048 | 237.3 | 222.5 |

Issue rates (`op.jsonl`, `op_rate.py` on A, giga lane-ops per second, 16 independent chains of one statement): fma 1234.5, fadd 1367.8, fmul 1368.1, iadd 1288.4, `ior` 613.5, `iand`+xor 778.2, shift+add 495.1, imul 495.3, u32-to-float 465.2, the old nibble decode 415.0. The in-place decode replaces the half-rate `or` and the shift with full-rate adds.

## Dispatch trace (`trace-*.txt`)

Profiling builds of `main` and of the six-change integration (fused routes forced on, so the same kernel list): granite decode runs 80 `QmmVecQ4MultiSubgroupBF16` and 81 `QmmVecQ4WordSubgroupBF16` dispatches per step in both, at 200.1 -> 189.7 and 109.4 -> 106.3 microseconds per dispatch (`MLX_OMARCHY_FUSED_GEMV=1 MLX_OMARCHY_FUSED_TRIO=1` on the integration `749902d`). Every dispatch is a Vulkan dispatch; none runs on the CPU.

## Device reopen

`reopen.py` passed before and after each bit-compare set and at the end of the session; no new GPU driver message in the kernel log during any run.

## Other parts

Correctness holds by construction on every device: the decoded float is the same bits. G13 and G14: not measured here. The speed effect depends on that GPU's integer add, or and shift rates; the commands for a future measurement are `python3 scripts/qmv_bits.py dump base.npz` on the base wheel, `python3 scripts/qmv_bits.py compare base.npz` on this change (exit 0 means bit-identical), `python3 scripts/gemv_sweep.py 7` on each, and `python3 scripts/ab.py <out.jsonl> 2 main <arm> -- decode_bench.py <repo> 64 4` for decode.

## Files

`host.txt`, `wheels.txt`, `battery.txt`, `bits-p3-vs-main.log`, `bits-selftest.log`, `engage-qmv-kernels.txt`, `perf-p3.jsonl`, `perf-p3-summary.txt`, `perf-p3-gemv.jsonl`, `op.jsonl`, `trace-*.txt`, `trace-*.run.log`, `reopen.log`, `scripts/` (the harness and probes), `SHA256SUMS`.
