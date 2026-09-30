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

## Addendum: packaged-module hardware verification on an M1 Max / T6001 host (2026-09-30)

The consumer path above was verified on the M1 / T8103 host for the Vulkan parts; the ANE worker gate then ran against a packaged DKMS `ane.ko` (`0.2.0.r14.g87f427f`) on a T6001 host. That run found two gate bugs, both fixed and unit-tested before the run passed:

- The ABI-1 profile accepted only `apple,t8103-ane`, but T6001 device-tree nodes carry the family compatible `apple,t6000-ane` (the `of_match` entry the driver binds). The pristine wheel refused the host at the DT stage, exactly as pre-registered: `device-tree ANE node does not match apple,t8103-ane for ABI 1 (M1 / T8103 / T6001).`
- The gate required `power/control` to be the literal `on`, but the packaged driver manages runtime PM (`auto`). A healthy, bound, `active` device was refused. The gate now opens `/dev/accel/accel0` before reading the power state — which resumes a runtime-PM device — and accepts `on` or `auto` control with `active` status (`runtime_pm_acceptable`, unit-tested).

After both fixes, under the host's lab locks and with bounded timeouts:

- A `DRM_IOCTL_VERSION` probe printed the compared values: `driver_name=ane version_major=1 expected_major=1 accepted=true`; the worker accepted the packaged module and its identity recorded `abi=1 dt_compatible=apple,t6000-ane driver_version=0.2.0.r14.g87f427f runtime_pm=auto/active driver_abi=1`.
- The repo's smallest real ANE workload — the h13 add-mul bundle adapted from the test fixture with the repo's own adapter (2 programs) — ran bit-exact (`exact_fp16=PASS`, expected fp16 `0x4600` on all 64 elements, 2 iterations) and shut down cleanly (`released_programs=2 process_released=true`).
- The same run as a non-root user who owns none of the files succeeded via the per-user ownership fallback (`ANE ownership uses per-user lock ...` under that user's `XDG_RUNTIME_DIR`), with identical bit-exact output and a clean shutdown receipt.
- Negative controls: the pristine wheel reproduced the pre-registered DT refusal; `MLX_OMARCHY_ANE_ABI=2` was refused with the ABI-2 lane error; `=99` was refused as unsupported. The driver-major mismatch branch (`requires major 1, driver reports major 2`) is unit-proven — the live driver cannot report another major and the code exposes no seam to fake the expected value.
- `scripts/mlx_provenance.py` reported `verified: match` for the loaded binaries against the installed wheel; wheel SHA-256 `a519c46a3fc240ff65974a09e5a1c07ba5aaad41995ad496360ec19ef5bd0a7a` (worker-gate doctest state: 19 cases / 75 assertions on both the build host and the target host).
- No dmesg ANE errors during the window; both per-user quarantine files were empty after the clean shutdowns.

Raw logs are in the private lab `artifacts/AbiVerify/` directory with `SHA256SUMS`; this addendum keeps only the public-safe results.
