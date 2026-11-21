# CBQ compile acceptance, pre-fix refusal baseline, and post-fix Apple GPU rows on the 16-inch M1 Max

Coverage for #524 at `e5d940eabda81460b6998e2c85ed40f8966e9498`, 2026-11-21 UTC. Device: Apple M1 Max (G13C C0), Honeykrisp, Mesa `26.3.0-devel (git-69bde15340)`. Kernel: `7.1.12-2-20261103.2-lab-ARCH`.

`tests/test_cbq_vulkan_translation.py`: 15 passed, zero skipped. All nine released bodies translated, compiled to SPIR-V, and passed validation. The near-match and five unsupported contracts refused as expected.

The runtime recheck used the Python extension built from the tested commit, not the installed wheel. The launcher asserted the loaded module path and selected only the Apple Vulkan driver. Each row ran `python -m tools.kernel_recheck.run --gpu --kernel NAME` in isolation. This run predates the #524 index-buffer fix (`3a4e7c99`): it is the pre-fix baseline, kept because the refusal reasons and per-row timings documented the ABI check that the fix later satisfied.

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

Pre-fix baseline (the #524 index-buffer fix at `3a4e7c99` came after these rows): the candidate recognized the pinned CBQ bodies but rejected each live call with `unsupported CBQ kernel ABI or launch`. The compile fixture did not exercise the live MLX wrapper. No kernel dispatched, so no output was compared against the CPU reference and no kernel time was measured. Harness elapsed includes subprocess startup and refusal; it is not GPU time. The fixed-head runs below supersede these rows.

## Fixed head: first dispatch of all nine kernels (pass)

Coverage for #524 at `3a4e7c999f981072d1c5711913bd92c483d95264`, 2026-11-21 UTC, same device and Mesa build, two boots (`ca7d5c52` first run, `5ebf207d` rerun). The #524 gate fix accepts the native index-type spellings, so every kernel now dispatches; none is refused.

`binary-provenance.txt` identifies the retained standalone extensions. These builds link MLX statically. The hashes were collected after the runs; they are not execution-time attestations.

Pass rules follow the upstream source (`k3_cbq.py`, HF `avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw`): the seven gather kernels keep the harness per-element rule; the two grad kernels, which upstream declares `atomic_outputs=True` and which carry no upstream numeric self-test, are judged at upstream's own bar, relative-L2 at most 1e-5 for fp32, with the per-element figure reported beside as information.

The binary outputs give maximum absolute errors of `0.00237640738` (`grad_d`) and `0.000386312502` (`grad_x`) against the f64 reference. The f32 and f64 CPU references differ. This comparison does not isolate the source of the GPU error.

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

`ref32-cpu-reference.txt` lists the elements over the per-element threshold and their float32 ULP distances. The worst relative errors occur at `(6, 266, 9)` for `grad_d` and `(0, 2440)` for `grad_x`. These elements differ by 2,307,888 and 920 ULPs from the rounded f64 reference. Neither element is the max-absolute-error index printed by the harness.

The `524-fixed-rows-ca7d5c52.txt` and `524-fixed-grad-rerun-5ebf207d.log` logs record the two boots. The rerun repeats the same reported summary across four samples per gradient kernel. The two `.npy` files preserve the rerun outputs used for the numeric comparison. `SHA256SUMS` covers the published files.
