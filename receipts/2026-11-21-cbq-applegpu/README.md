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

The compile log, the runtime rows, and all nine child logs are included. Machine-local paths in the logs are redacted. `SHA256SUMS` covers the published evidence files.
