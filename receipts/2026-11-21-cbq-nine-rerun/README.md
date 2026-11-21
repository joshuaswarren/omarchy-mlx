# CBQ nine-kernel GPU rerun at `f0f6563fc` (2026-11-21)

Fresh validation of the bounded exact-match CBQ port (#524, commit
`6f5e6a827`) at main `f0f6563fc3c8a8e2a579ccc0cac3758e782962e4` on a
13-inch M1 Mac (Apple M1, T8103), run through the idle-guard GPU queue as
a `class=perf` ticket. Extends the earlier GPU pass receipted at
`3a4e7c99` on the M1 Max (#538, `receipts/2026-11-21-cbq-applegpu/`):
this run measures the merged head on a second chip line.

## Build

`class=build` ticket `864d3bc5…` (guard record in `guard-logs.txt`)
ran `scripts/build-wheel.sh` at `f0f6563fc` with `DEV_RELEASE=1`.

- Wheel: `mlx_omarchy-0.32.4.dev202611211520+f0f6563-cp314-cp314-linux_aarch64.whl`
- Wheel SHA-256: `5ed654604b729a7ee19bc9680a7b1974a331c46cedb739b42a2946964b852594`
- `libmlx.so` SHA-256: `86bb441ff85aeb6615369cbebcd5587b3a92aebe3879503e8b1e4040eb3a4133`
- Kernel: `7.3.0-2-20261110.1-lab-ARCH`
- Chip: Apple M1 (T8103)
- Host: lab Mac (identifier withheld)

## Compile acceptance (control host, no GPU)

`CBQ_TRANSLATE_DUMP=…/omarchy_custom_kernel_translate_dump pytest
tests/test_cbq_vulkan_translation.py`: **17 passed**. All nine released
bodies translate and compile to valid SPIR-V; the `as_type<char4>`
near-match still refuses, and the six unsupported ABI/launch mutations
refuse with `unsupported CBQ kernel ABI or launch`.

## GPU recheck

`class=perf` ticket `681f2ada…` ran `run-recheck.sh`: a fresh venv from
the wheel above, then per kernel
`python -m tools.kernel_recheck.run --gpu --kernel NAME`.

| Kernel | Result | Max difference |
|---|---|---:|
| `cbq_gather_mm` | pass | 0.00657895 |
| `cbq_gather_mm_v2` | pass | 0.00657895 |
| `cbq_gather_mm_v3` | pass | 0.00657895 |
| `cbq_gather_mm_v4` | pass | 0.00657895 |
| `cbq_gather_mm_v3_situ` | pass | 0 |
| `cbq_gather_mm_v4_situ` | pass | 0 |
| `cbq_gather_mm_glu` | pass | 0 |
| `cbq_grad_d` | **wrong** | rel 0.2098, 33/65536 bad |
| `cbq_grad_x` | **wrong** | rel 1.037e-4, 1/4096 bad |

Summary: **7 PASS / 2 WRONG / 0 REFUSED / 0 FAIL** under the harness's
per-element `("rel", 1e-4), atol=0` gate. `cbq_grad_d`: 33 of 65536
outputs over the gate (worst absolute-diff element -67541.72 vs
-67541.71637359262; max relative error 0.2098 on a near-zero element).
`cbq_grad_x`: 1 of 4096 over the gate (rel err 1.037e-4, margin 4e-6).

Both verdicts reproduced bit-identically across a separate 3-rep
`class=perf` ticket (`2420ab84…`, `gradreps-out.txt`), and — verified
against `receipts/2026-11-21-cbq-applegpu/` — the **identical** signature
(same bad counts, indices and values) was recorded on M1 Max at
`3a4e7c99` under #538. The numerics are therefore chip-independent and
deterministic. On M1 Max the kernels were judged under upstream's
declared bar for `atomic_outputs` kernels, relative-L2 ≤ 1e-5, and
recorded rel-L2 `3.64844704e-08` / `3.57092598e-08` — a pass with
~270x margin, with the per-element figures reported as information.

So this run does not show a regression on M1: it shows that the two
`atomic_outputs` gradient kernels sit outside the harness's per-element
gate while comfortably inside upstream's rel-L2 criterion, on both
chips. Which gate is authoritative for `atomic_outputs` rows is tracked
in issue #564.

Provenance: `scripts/mlx_provenance.py` output and the full per-kernel
RESULT lines are in `recheck-out.txt` in this directory; the three-rep
repro is `gradreps-out.txt`; guard-side execution records are
`guard-logs.txt`. The run used the Vulkan/Borealis backend end to end;
no CPU tensor path ran.
