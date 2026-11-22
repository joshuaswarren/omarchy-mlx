# 2026-11-22 — libane ABI-2 repin proposal (jwm2 dev_id-0 device-open failure)

## Defect

omarchy-mlx 0.7.49-2 on jwm2 (T6021, kernel `7.3.0-2-20261110.1-lab-ARCH`)
cannot open the ANE: the parakeet worker's `__ane_init(model, 0)` fails
with `failed to find device with dev_id 0` while kernel-side ANE rows keep
succeeding.

## Root cause (every claim tied to a hash or live output)

| Fact | Evidence |
| --- | --- |
| Loaded driver: `updates/ane_t6021.ko` v0.5.10, srcversion `84ED10E0AE8039278BDCED0`, sha256 `339629a1…`, placed 2026-11-12 23:24, depmod-shadows the in-tree module | `modinfo`, `/proc/modules`, `stat` on jwm2, 2026-11-22 |
| That driver registers `.major = ANE_ABI_M2_MAJOR` (2), `.name = "ane"` | omarchy-ane `ane/src/uapi/drm/ane_accel.h` (`ANE_ABI_M2_MAJOR 2`), `ane/t6021/ane_t6021_rtclient_main.c` (`27e996a`, 2026-09-29, "installed ABI-2 driver port") |
| Pinned libane: `libane-strict.so` sha256 `d06222a8…`, built from omarchy-ane `df5c553` (2026-09-22) | `parakeet-runtime-pin.json` (identical at v0.7.41 and v0.7.49; pin json history ends 2026-09-24 `5a7b371e32`) |
| That libane accepts only `version_major == ANE_ABI_MAJOR` (1) | omarchy-ane `df5c553:libane/ane.c:289` (`if (!version.name_len || version.version_major != ANE_ABI_MAJOR)`) |
| Live reproduction, real device | `__ane_init(island-pv/program-0.anec, 0)` → `LIBANE: ERR: failed to find device with dev_id 0`; red test `overlay/tests/omarchy/ane/test_libane_pin_abi2.py` on jwm2, 2026-11-22 |

ABI 2 landed in the driver seven days after the pin was cut, and the pin
has never moved. omarchy-mlx 0.7.49 did not change anything ANE-side
(`git diff v0.7.41 v0.7.49` over the ANE surface touches only
`mlx_omarchy_parakeet.py`); installing 0.7.49-2 merely re-shipped the
stale pin. The M1 fleet is unaffected (ABI 1 still accepted), which is why
jwm1 parakeet rows keep passing on the same release.

## Fix (legal path per the 2026-10-02 libane-pin receipt)

Rebuild, then move the pin in the same change:

1. Rebuild `libane-strict.so` from omarchy-ane main (≥ `8a4379e` which adds
   the libane ABI-2 T6021 backend; `is_ane_device` there accepts majors 1
   and 2), using the receipt's byte-reproducible recipe
   (`gcc -shared -fPIC -O2 -Wall`, no `LIBANE_CONFIG_STRICT_BIND`), and
   record the commit in the provenance string.
2. Update `parakeet-runtime-pin.json`: new `libane-strict.so` sha256 +
   provenance; run `scripts/verify_runtime_assets.py` on the wheel tree
   before `pip wheel` and on the staged venv in `install.sh --system`
   (both hooks already exist from the LibanePin fix).
3. Release 0.7.50; deploy to the M2 hosts (jwm2). M1 hosts can take it or
   stay — ABI 1 remains accepted either way.

Rejected alternatives:

- Downgrade the jwm2 driver to an ABI-1 ane_t6021: would break the live
  v0.5.10 h14/h15/h16 program rows (journal, 2026-11-21/22) and the
  in-tree lab modules already report major 2.
- Pin to an intermediate commit between `df5c553` and `8a4379e`: buys
  nothing; latest main is the simplest byte-reproducible target.

## Guard

`overlay/tests/omarchy/ane/test_libane_pin_abi2.py` (this branch) is red
on T6021 today and green after the repin; it also stays green on M1 hosts
and skips cleanly on x86 CI and on hosts without an accel node. Follow-up
(optional): extend `scripts/verify_runtime_assets.py` to record the
driver ABI majors the pinned libane accepts in the pin json, so the next
driver ABI bump fails the wheel build instead of the fleet.
