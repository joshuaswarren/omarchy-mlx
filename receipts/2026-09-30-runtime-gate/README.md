# Runtime gate

This receipt covers ABI acceptance, ANE lock access, and Honeykrisp selection.

## Implementation

The ANE DRM driver exports `ANE_ABI_MAJOR` as its DRM driver major. The worker reads `DRM_IOCTL_VERSION.version_major` from `/dev/accel/accel0` and compares it with the runtime ABI. A missing value or mismatch fails with both ABI values. The worker does not accept a module version string as an ABI pin.

The shared ANE lock directory is sticky and world-writable. Lock and quarantine files use mode `0666`. The runtime repairs a mode-mismatched file when it owns it. If a shared file cannot be opened for mode repair, the runtime reports and uses its per-user lock under `XDG_RUNTIME_DIR`. Device-node permissions remain the driver and udev configuration's responsibility.

Before loading Vulkan, the backend selects the Honeykrisp ICD. It accepts an explicit `VK_DRIVER_FILES` or `VK_ICD_FILENAMES` value only when that value includes Honeykrisp. An excluding value produces an error. `mx.device_info()` and `mlx-omarchy-info` report the selected ICD path, driver name, driver info, and Mesa git SHA. `MLX_OMARCHY_EXPECTED_HK_SHA` enforces the expected SHA when set; when unset, the runtime reports identity without rejecting it.

## Verification

A private wheel built with `glslc` on an Apple M1 / T8103 host. The wheel size was 416,222,746 bytes and its SHA-256 was `bde3934e7259f8b8efb4cea8432b548db8439dc52e636d1d8ecf89fa9825dac8`. `scripts/mlx_provenance.py` reported `verified: match` for the installed wheel and both loaded binaries.

The focused worker-gate doctest passed 17 cases and 64 assertions on the build host. On the M1, `omarchy_runtime_tests` passed 41 cases and 22,694 assertions. `omarchy_error_contract_tests` passed 3 cases and 14 assertions. No cases were skipped.

The M1 info tool reported device `Apple M1 (G13G B1)`, driver `Honeykrisp`, driver info `Mesa 26.3.0-devel (git-f04cf2e97d)`, SHA `f04cf2e97d`, ICD `/usr/share/vulkan/icd.d/asahi_icd.json`, Vulkan 1.4.359, kernel `7.1.13-3-2-ARCH`, device-tree compatibles `apple,j293 apple,t8103 apple,arm-platform`, and ANE module version `5ecff86`. The expected-SHA override was unset.

A real GPU matmul returned the exact expected 4×4 result in the inherited environment and under `env -i PATH=/usr/bin:/bin`. Both runs reported Honeykrisp, the same ICD path and SHA, and matching wheel provenance. The scrubbed process had neither Vulkan ICD environment variable set before importing MLX.

```text
[[56, 62, 68, 74],
 [152, 174, 196, 218],
 [248, 286, 324, 362],
 [344, 398, 452, 506]]
```

Commands ran under `flock /tmp/m1-gpu.lock` with bounded timeouts. The M1 test, info, provenance, and matmul logs are in the local `artifacts/RuntimeGate/20260930-jwm1-runtime-gate/` directory. `SHA256SUMS` covers those logs. Raw logs are not committed because they contain private host paths; this receipt contains the public-safe results.
