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

## Correction, 2026-09-30

Defect found on another host: an explicit `VK_DRIVER_FILES` value naming a
missing Honeykrisp JSON (for example `asahi_icd.json` where the installed
file is `asahi_icd.aarch64.json`) was accepted by the ICD selection, the
Vulkan loader then failed, and the process silently continued with the CPU
default device.

Two fixes:

1. ICD selection now verifies that an explicit override entry exists. A
   Honeykrisp-named entry whose file is missing fails initialization with
   `Honeykrisp ICD selection refused: user Vulkan ICD JSON does not exist:`
   and the exact path. A value that excludes Honeykrisp is refused as
   before.
2. The default-device selection no longer degrades silently. In release
   builds, a failed GPU backend initialization raises the recorded
   compatibility error; the CPU device is never chosen implicitly. Debug
   builds keep the CPU fallback for development hosts without Apple GPUs.

Unit tests cover both: a doctest refuses an injected missing override path
and checks the exact message, and a doctest checks the release refusal,
the debug fallback, and the GPU selection for the default-device policy.
On an x86 host with lavapipe installed, the info tool refuses a missing
Honeykrisp override naming the exact path, refuses a lavapipe override,
and reports the missing Honeykrisp JSON with no override.

Packaged ICD contract (v0.7.7 recipe lane), same day: with no override,
the packaged `$prefix/vulkan/honeykrisp_icd.aarch64.json` is preferred
over any stock system ICD, and when it is selected the expected SHA comes
from the packaged `mesa-git-sha` file; an explicit env SHA wins, and an
env-selected ICD never inherits the packaged SHA. Device info reports the
ICD source (`packaged`, `override`, `search`), the expected SHA, and its
source (`env`, `packaged file`, or none). Prefix `/usr/lib/omarchy-mlx`
is a build-time constant with an `OMARCHY_MLX_SYSTEM_PREFIX` runtime
seam; the names live in `serve/mlx_omarchy_paths.py` and generated
`packaging/paths.sh`.

The packaged-ICD and ICD-refusal confirmation ran on jw16 (M1 Max, G13C,
T6001, kernel `7.1.13-3-2-ARCH`) because jwm1 was reserved by `H176`
and `MesaParity`; the existing preserved wheel built at `31af03eac`
(`/var/tmp/omarchy-mlx-wheels/mlx_omarchy-0.32.3.dev202610010514+31af03e-cp314-cp314-linux_aarch64.whl`,
sha256 `c49ecca0950659f445fd86e873a9527b3fac6920780f18020f9075df1ec2d6b7`)
was reused. One `gpuwin` slice under `flock /tmp/m1-gpu.lock`, announced
to `w72:p1`, restored with `llm-inference.service` `active` and a real
completion probe:

- `omarchy_runtime_tests` 41 cases / 41 passed / 22,694 assertions / 0
  failed. `omarchy_error_contract_tests` 3 cases / 3 passed / 14
  assertions / 0 failed. `scripts/mlx_provenance.py` reported
  `verified: match`, `version_match: true`.
- The M1 Max info tool reported device `Apple M1 Max (G13C C0)`, driver
  `Honeykrisp`, driver info `Mesa 26.3.0-devel`, ICD
  `/usr/share/vulkan/icd.d/asahi_icd.aarch64.json`, Vulkan 1.4.359,
  `architecture=honeykrisp`, ANE module `0.2.0.r14.g87f427f`. The Mesa
  build on this box carries no `git-` token in `driver_info`, so
  `driver_sha=""` and the packaged `mesa-git-sha` file cannot carry a
  live-read SHA on this hardware (expected-sha enforcement is exercised
  below via the wrong-value case).
- A 4×4 `f32` matmul returned the matrix the existing receipt records
  (`[[56, 62, 68, 74], [152, 174, 196, 218], [248, 286, 324, 362],
  [344, 398, 452, 506]]`) in the inherited environment and under
  `env -i PATH=/usr/bin:/bin`. MLX matmul does not accept integer
  operands; the prior int32 smoke row is reproduced here as the f32
  matrix.
- The fake packaged tree at `/var/tmp/runtimegate-packaged/vulkan/honeykrisp_icd.aarch64.json`
  (system ICD JSON copied verbatim, `mesa-git-sha` populated from the
  live driver read; `OMARCHY_MLX_SYSTEM_PREFIX` redirected the packaged
  ICD path) flipped selection source from `search` to `packaged`
  (`mlx-omarchy-info --json` reports `icd_source=packaged`,
  `icd_path=/var/tmp/runtimegate-packaged/vulkan/honeykrisp_icd.aarch64.json`)
  and the 4×4 matmul returned the same matrix through that pointer.
- Wrong SHA in the packaged `mesa-git-sha` (`000000000000000`) refused
  with `RuntimeError: [omarchy] refusing CPU tensor fallback in a release
  build; the GPU backend is unavailable: Honeykrisp Mesa git SHA mismatch:
  expected 000000000000000, found unavailable` — both pre-registered
  errors (file SHA mismatch → CPU refusal) in one message. Exit code 1.
- `MLX_OMARCHY_EXPECTED_HK_SHA=0123456789abcdef` refused with the same
  prefix and `SHA mismatch: expected 0123456789abcdef, found unavailable`
  (inherited and `env -i PATH=/usr/bin:/bin`, exit 1).
- `VK_DRIVER_FILES=/var/tmp/runtimegate-packaged/vulkan/nonexistent_asahi_icd.json`
  refused with `RuntimeError: ... Honeykrisp ICD selection refused: user
  Vulkan ICD JSON does not exist: /var/tmp/runtimegate-packaged/vulkan/nonexistent_asahi_icd.json`.
  Exit code 1.

The M1 confirmation on jwm1 (`G13G`) remains pending. Raw logs are in the
private lab `artifacts/IcdConfirm/20261001-jw16-icd-confirm/` directory
with `SHA256SUMS`; only the public-safe results are recorded here.
