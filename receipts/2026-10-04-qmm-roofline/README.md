# QMM prefill roofline and dequantization variants

Status: **landed default-on 2026-10-04** (G4-route shared B-tile stride
padded; kill switch `MLX_OMARCHY_QMM_LDSPAD=0`). The T6021/G14C evidence
below is complete for the 2B/4B/9B grid; the M1/G13 model-level
confirmation remains owned by the jwm1 lane.

## Landing decision

The lane's own promotion bar was >=1.5% on every cell plus the
bit-equality doctest. T6021 met it on most cells but the 4B pf1024 cell
did not (valid pairs -0.61% to +2.15%). Landing default-on anyway is an
orchestrator decision (QmmPeak lane, 2026-10-04), recorded here with its
reasoning: every candidate output kept identical activation and logits
digests on both silicon variants, all 30 measured jwm1/G13G blocks were
positive (+0.9% there, +2.1-3.0% on T6021), the only non-positive
samples were that one noisy 4B pf1024 cell, and the numerics risk is
zero by construction (the padding changes no value and no accumulation
order). Cost is 256 B more shared memory per workgroup, inside every
supported budget. The 1.5% bar was waived for this landing, not met.

## Question

Measure, for the 2B/4B/9B model QMM shapes at M=512 and 1024, achieved TFLOPS for the shipped quantized matmul, an ordinary cooperative-matrix matmul, and a matmul over pre-dequantized bf16 weights. Then test whether chunk-level dequantization or a padded shared-memory row stride improves prefill without changing output bits.

## CPU-side changes

Branch `agent/qmmpeak-roofline`, based on `origin/main` at `b59f4bef7`:

- `abe9a7ddf`: adds env-gated `CHUNK_DEQ`, `LDS_PAD`, and combined QMM kernel variants, plus exact-equality doctest arms. Both knobs default off.
- `11c313f56`: corrects routing precedence so an explicit twin knob selects its arm only when the default G4 raster route applies and no explicit raster control overrides it.

The dev-box compile built `libmlx.a`; the four QMM shader configurations compiled with `glslc`, and the staged diagnostic wheel reported `0.32.4.dev202610041808+diag.11c313f5`. The on-device T6021 matmul-family test target was built from the current branch and its exact-equality doctest passed: 1 test case, 1,920/1,920 assertions, 0 failures, covering all eight K,N test shapes and M in {17..2047 odd}. This verifies candidate bit equality on T6021, not M1/G13. Doctest log SHA-256 `4fef142b06b46f49ac2c053bfd0ad9076dfea23d36d065befdda27ed84e5e24f`; raw log archived with the other T6021 measurements.

## Offline SPIR-V comparison

Compiled on the x86 dev box using `glslc -O --target-env=vulkan1.3`. These counts describe SPIR-V only; they do not reveal Honeykrisp/NIR/AGX registers, occupancy, spills, or runtime barriers. Raw output archive: `~/.local/share/apple-silicon-lab/artifacts/QmmPeak/qmmpeak-spv-devbox.tar.gz`, SHA-256 `72236c3bd53a35e3f0b7e4ebc0b76367e3aca0f93269537c0448f82b9a79eb3e`.

| Arm | Shared B staging | SPIR-V words | Static ops | `OpControlBarrier` | coop loads | coop mul-add |
|---|---:|---:|---:|---:|---:|---:|
| shipped G4 control | 2,048 B | 4,391 | 1,096 | 4 | 2 | 1 |
| CHUNK_DEQ | 8,192 B | 4,495 | 1,122 | 4 | 2 | 1 |
| LDS_PAD | 2,304 B | 4,399 | 1,098 | 4 | 2 | 1 |
| CHUNK_DEQ + LDS_PAD | 9,216 B | 4,503 | 1,124 | 4 | 2 | 1 |

The chunk arm removes barriers from the inner MAC walk in source, but the static SPIR-V barrier count remains four because those sites are inside runtime loops and branches. Static counts do not report dynamic barrier executions. Its shared-memory footprint is 4x the control arm; target-compiler occupancy cost remains unmeasured. Do not treat these offline counts as evidence of a win.

## Roofline data

T6021 (G14C, diag wheel 0.32.4.dev202610041921+diag.11c313f5) produced 40 shape/M cells with matching provenance. A first attempt stopped at the bfloat16 digest conversion and is excluded; the corrected driver widened bf16 output exactly to f32 for hashing, then completed all 40 cells. The dequantized-bf16 and plain-f32 baselines use ordinary `mx.matmul`; this run did not force or verify a standalone cooperative-matrix-only path, so they are practical backend baselines, not a separately measured coopmat ceiling. These results are T6021-only; they do not establish G13 performance or a default-on win.

| Model | K,N | M | QMM TFLOPS (512 / 1024) | dequantized bf16 matmul TFLOPS (512 / 1024) | plain f32 matmul TFLOPS (512 / 1024) | provenance / status |
|---|---|---|---|---|---|---|
| 2B | 2048, 4096 | 512 / 1024 | 6.16 / 6.47 | 2.35 / 2.85 | 1.48 / 1.82 | T6021 (G14C), provenance matched; G13 pending |
| 2B | 2048, 512 | 512 / 1024 | 3.52 / 3.81 | 1.72 / 2.86 | 0.80 / 1.53 | T6021 (G14C), provenance matched; G13 pending |
| 2B | 2048, 16 | 512 / 1024 | 0.13 / 0.23 | 0.06 / 0.13 | 0.03 / 0.06 | T6021 (G14C), provenance matched; G13 pending |
| 2B | 2048, 2048 | 512 / 1024 | 5.24 / 6.03 | 2.84 / 2.48 | 1.54 / 1.97 | T6021 (G14C), provenance matched; G13 pending |
| 2B | 2048, 6144 | 512 / 1024 | 6.49 / 6.67 | 2.87 / 3.11 | 1.65 / 1.80 | T6021 (G14C), provenance matched; G13 pending |
| 2B | 6144, 2048 | 512 / 1024 | 5.42 / 6.13 | 2.81 / 2.38 | 1.41 / 1.45 | T6021 (G14C), provenance matched; G13 pending |
| 2B | 2048, 248320 | 512 / 1024 | 7.09 / 6.97 | 3.17 / 3.19 | 1.81 / 1.79 | T6021 (G14C), provenance matched; G13 pending |
| 4B | 2560, 4096 | 512 / 1024 | 6.09 / 6.35 | 2.35 / 2.84 | 1.46 / 1.79 | T6021 (G14C), provenance matched; G13 pending |
| 4B | 2560, 1024 | 512 / 1024 | 4.01 / 5.08 | 2.83 / 2.90 | 1.50 / 1.53 | T6021 (G14C), provenance matched; G13 pending |
| 4B | 2560, 2560 | 512 / 1024 | 5.90 / 6.19 | 2.44 / 2.76 | 1.76 / 1.55 | T6021 (G14C), provenance matched; G13 pending |
| 4B | 2560, 9728 | 512 / 1024 | 6.51 / 6.67 | 3.04 / 3.18 | 1.82 / 1.95 | T6021 (G14C), provenance matched; G13 pending |
| 4B | 9728, 2560 | 512 / 1024 | 6.18 / 6.29 | 2.38 / 2.75 | 1.70 / 1.45 | T6021 (G14C), provenance matched; G13 pending |
| 4B | 2560, 151936 | 512 / 1024 | 6.87 / 6.88 | 3.12 / 3.19 | 1.80 / 1.79 | T6021 (G14C), provenance matched; G13 pending |
| 9B | 4096, 8192 | 512 / 1024 | 6.62 / 6.83 | 3.14 / 3.24 | 1.78 / 2.00 | T6021 (G14C), provenance matched; G13 pending |
| 9B | 4096, 1024 | 512 / 1024 | 4.18 / 5.27 | 2.89 / 2.97 | 1.56 / 1.44 | T6021 (G14C), provenance matched; G13 pending |
| 9B | 4096, 32 | 512 / 1024 | 0.25 / 0.47 | 0.11 / 0.23 | 0.05 / 0.11 | T6021 (G14C), provenance matched; G13 pending |
| 9B | 4096, 4096 | 512 / 1024 | 6.28 / 6.52 | 2.79 / 3.04 | 1.49 / 1.78 | T6021 (G14C), provenance matched; G13 pending |
| 9B | 4096, 12288 | 512 / 1024 | 6.81 / 6.95 | 3.12 / 3.19 | 1.80 / 1.88 | T6021 (G14C), provenance matched; G13 pending |
| 9B | 12288, 4096 | 512 / 1024 | 6.25 / 6.42 | 2.86 / 3.19 | 1.43 / 1.75 | T6021 (G14C), provenance matched; G13 pending |
| 9B | 4096, 248320 | 512 / 1024 | 7.11 / 7.03 | 3.08 / 3.06 | 1.76 / 1.76 | T6021 (G14C), provenance matched; G13 pending |

## Variant A/B

On T6021/G14C, LDS_PAD preserved activation and logits digests in every candidate comparison. The 2B model completed five clean alternating pairs at both pf512 (W2: +2.18-2.98%) and pf1024 (W3: +2.12-2.35%). 4B pf512 completed five clean pairs in W4R2 (+2.2-2.6%; exact digests); earlier W4 and W4R contributed one and three clean pairs. 4B pf1024 produced three clean pairs in W7R (+2.11-2.27%) and four in W7R2; W7R2 valid pair gains ranged from -0.61% to +2.15%, so that cell does not show a stable >=1.5% win across pairs. 9B pf512 (W10) completed five clean pairs (+2.48-2.61%); pf1024 (W13 and W13R) produced five clean pairs overall (+2.04-2.62%). W7R2's first pair was rejected for load gate; no red-gate sample is counted. All valid measurements passed matching provenance. This is T6021/G14C-only evidence; M1/G13 remains required before default-on or generalizing performance. CHUNK_DEQ remained slower by 16.1% at pf512 and 16.5-16.7% at pf1024 on valid load-gated 2B pairs. CHUNK_DEQ+LDS_PAD remained slower by 26.0-26.3% at pf512; pf1024 had only two valid gated pairs, both about 28.7% slower before load became red.

W2 SHA-256: 3253ff6914af2875c3ec89ecc03dccd00f9cf80397511318eac451335725101e. W3 SHA-256: 72d00dac321f19b13fc14419b2bf224ab0679f836d55ff67de07b86c31e80527. W4R2 SHA-256: 5da8705d1ec8681269c868152917e37a01f29ad90614ddbf344257cb2b3ef101 (stderr a77683925e729a1a536c04e4d3ddc43a66129364d5d45bfea94ed4d4306f74b2). W7R2 SHA-256: b28200bf047c43cd4a6e066f1f56859a440513735b21b0aeb66e56db72a3018e (stderr 30c23f9b8fea937d90314e6db94abbf51f4ce6e278462012f01dc93b44796641). Doctest log SHA-256: 4fef142b06b46f49ac2c053bfd0ad9076dfea23d36d065befdda27ed84e5e24f. Raw logs are archived at macstudio:/Volumes/Turbo/oracle-mint-scratch/laptop-archive-20261004/QmmPeak-T6021/.

## T6021 verification of the default-on landing

After inverting the knob, the T6021 test build ran one load-gated GPU
window (`battery-defon`, 2026-10-04):

- Twins doctest with the new semantics (default arm = padded route,
  baseline pinned with `NO_RASTER=1` + `LDSPAD=0`): 1/1 case,
  1,920/1,920 assertions, 0 failures.
- Full `omarchy_matmul_family_tests`: 23/23 cases, 82,942,383
  assertions, 0 failures.
- `omarchy_capability_sim_tests` (no profile arg): profile-registry unit
  leg OK; the binary then refuses without an explicit profile name, so
  the per-profile battery matrix stays with the M1/G13 qualification
  window, as do the remaining standing-battery suites (runtime,
  primitive, fast ops, SDPA, GDN and the rest) - the M2 block ended
  before they could run, and the orchestrator limited the M2 to plain
  tickets.

Logs SHA-256: jsonl `f7fcb3952a6ce78526668700badda5bc3cdfffac770a1848d4ab586c2a9f3f3b`,
stderr `d2c35a5070cce15270f23cf5938af159a713f3b0141ae839671926604901cf33`,
archived with the other T6021 evidence.

## M1/G13 handoff

1. The jwm1 lane runs the targeted LDS_PAD A/B grid on M1 per model and
   prefill cell; preserve load/PSI gates, provenance, and exact output
   digests. Note the knob is now default-on: the control arm must set
   `MLX_OMARCHY_QMM_LDSPAD=0` explicitly.
2. Run the standing M1 battery there (the suites this M2 block could not
   cover, plus the capsim profile matrix).
3. Record M1 results in this receipt; the landing stands on the T6021
   evidence and the G13G block positives unless M1 contradicts.

Scratch archive: macstudio:/Volumes/Turbo/oracle-mint-scratch/laptop-archive-20261004/QmmPeak/qmmpeak-jw16.tgz, SHA-256 8cfb5d664021f6c835b009defca3cbc9e869fe64abda3454a27cf67062ae3254. Archive checksum and tar listing were read back on the archive host. T6021 raw measurement logs are archived separately in the QmmPeak-T6021 subdirectory; their hashes are listed above.
