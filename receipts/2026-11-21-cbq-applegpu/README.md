# CBQ compile acceptance and runtime refusals on the 16-inch M1 Max

Coverage for #524 at `e5d940eabda81460b6998e2c85ed40f8966e9498`, 2026-11-21 UTC. Device: Apple M1 Max (G13C C0), Honeykrisp, Mesa `26.3.0-devel (git-69bde15340)`. Kernel: `7.1.12-2-20261103.2-lab-ARCH`.

`tests/test_cbq_vulkan_translation.py`: 15 passed, zero skipped. All nine released bodies translated, compiled to SPIR-V, and passed validation. The near-match and five unsupported contracts refused as expected.

The runtime recheck used the Python extension built from the tested commit, not the installed wheel. The launcher asserted the loaded module path and selected only the Apple Vulkan driver. Each row ran `python -m tools.kernel_recheck.run --gpu --kernel NAME` in isolation.

| Kernel | Runtime status | Max difference | Harness elapsed (s) |
|---|---|---:|---:|
| `cbq_gather_mm` | Refused | n/a | 0.2 |
| `cbq_gather_mm_v2` | Refused | n/a | 0.2 |
| `cbq_gather_mm_v3` | Refused | n/a | 0.2 |
| `cbq_gather_mm_v4` | Refused | n/a | 0.2 |
| `cbq_gather_mm_v3_situ` | Refused | n/a | 0.2 |
| `cbq_gather_mm_v4_situ` | Refused | n/a | 0.2 |
| `cbq_gather_mm_glu` | Refused | n/a | 0.2 |
| `cbq_grad_d` | Refused | n/a | 0.2 |
| `cbq_grad_x` | Refused | n/a | 0.2 |

All nine stay classified as **refused**. The candidate recognized the pinned CBQ bodies but rejected each live call with `unsupported CBQ kernel ABI or launch`. The compile fixture does not exercise the live MLX wrapper. No kernel dispatched, so no output was compared against the CPU reference and no kernel time was measured. Harness elapsed includes subprocess startup and refusal; it is not GPU time. The check that rejected the live wrapper has not been isolated.

## Fixed head: first dispatch of all nine kernels (pass)

Coverage for #524 at `3a4e7c999f981072d1c5711913bd92c483d95264`, 2026-11-21 UTC, same device and Mesa build, two boots (`ca7d5c52` first run, `5ebf207d` rerun). The #524 gate fix accepts the native index-type spellings, so every kernel now dispatches; none is refused.

Pass rules follow the upstream source (`k3_cbq.py`, HF `avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw`): the seven gather kernels keep the harness per-element rule; the two grad kernels, which upstream declares `atomic_outputs=True` and which carry no upstream numeric self-test, are judged at upstream's own bar, relative-L2 at most 1e-5 for fp32, with the per-element figure reported beside as information.

Reason, two lines: float32 accumulation order under atomics differs from the CPU reference's fixed order, so a few near-zero elements exceed a pure per-element relative bound while staying sub-ulp at their magnitude. The float32 and float64 CPU references agree exactly, so the deltas come from the atomic accumulation, not reference precision.

| Kernel | Runtime status | Max difference | rel-L2 (bar 1e-5) | Harness elapsed (s) |
|---|---|---:|---:|---:|
| `cbq_gather_mm` | Pass | 0.0066 | n/a | 0.3 |
| `cbq_gather_mm_v2` | Pass | 0.0066 | n/a | 0.3 |
| `cbq_gather_mm_v3` | Pass | 0.0066 | n/a | 0.3 |
| `cbq_gather_mm_v4` | Pass | 0.0066 | n/a | 0.3 |
| `cbq_gather_mm_v3_situ` | Pass | 0 | n/a | 0.4 |
| `cbq_gather_mm_v4_situ` | Pass | 0 | n/a | 0.3 |
| `cbq_gather_mm_glu` | Pass | 0 | n/a | 0.4 |
| `cbq_grad_d` | Pass at rel-L2 3.648e-08 | 0.2098 rel (33/65536 elements over 1e-4) | 3.648e-08 | 2.8 |
| `cbq_grad_x` | Pass at rel-L2 3.571e-08 | 1.04e-4 rel (1/4096 elements over 1e-4) | 3.571e-08 | 0.3 |

Both flagged elements are sub-ulp at magnitude (0.46 and 0.75 float32 ulp). The rows reproduce exactly across the two boots and across four samples per kernel (first dispatch plus three probes): same bad-element indices, same values, same maxdiff — deterministic per compiled kernel, not run-to-run spread. Evidence files: `524-fixed-rows-ca7d5c52.txt` (first-run rows), `524-fixed-grad-rerun-5ebf207d.log` (rerun with rel-L2 and probes), `cbq_grad_d.out0.npy` and `cbq_grad_x.out0.npy` (saved outputs), `ref32-cpu-reference.txt` (float32 vs float64 CPU reference agreement, with a port proof against the shipped reference). Machine-local paths and framework lines are redacted. `SHA256SUMS` covers the published evidence files.

The compile log, the runtime rows, and all nine child logs are included. Machine-local paths in the logs are redacted. `SHA256SUMS` covers the published evidence files.
