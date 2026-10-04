# 2026-10-04: T6021 (M2 Max) GPU base pstate, opt-in overlay `gpu-pstate-t6021`

Lab entry: `jwm1-parity/20261004T033000Z-m2-gpu-base-pstate-offline-build-h253.md`
(raw data and scripts in the lab store, `artifacts/jwm1-parity/h253/`, sealed
with SHA256SUMS). Owner: w71 (jwm1 lane), M2 owner w73, repo owner w6Z.

## Question

Does the T8103/T6001 finding (the AGX firmware starts a burst from base pstate 1
and ramps, so a request after a 100 ms to 2 s pause pays 4 to 9 ms; setting the
base pstate removes it at no cost) hold on T6021, and which value?

## Answer

Yes for base pstate **6** (1236 MHz). **No for 8**, the top state (1398 MHz),
which is the value the T8103 and T6001 overlays use because it is their top.

| arm | after-pause first token (100/500/2000 ms) | sustained decode / prefill | idle |
|---|---|---|---|
| stock (two boots) | 87.2-91.0 / 88.3-89.5 / 93.0-94.3 ms | 109.6 tok/s / 1609 tok/s | 14.9-15.0 W |
| base 8 | 87.2 / 87.4 / 91.1 ms | -1.9 % / -3.3 %; 20 ms-gap request +2.5 ms | 15.2 W |
| base 6 | 84.9 / 84.7 / 87.6 ms | equal to stock within 0.1 % | 15.1 W |

Digests identical in every arm. Details: `docs/gpu-base-pstate-t6021.md`.

## How it was applied (no packaged path was used on the M2)

The M2 boots a lab m1n1 stage from the ESP, not a boot.bin built by
update-m1n1 from this package. The sealed boot.bin (sha256 62ba3010...) has the
update-m1n1 layout (m1n1 stage, 110 raw DTBs, gzip U-Boot). The images were built
offline by replacing the j414c DTB with an `fdtoverlay`-merged copy of itself;
each image differs from the sealed file in exactly one byte (offset 0x27fda5,
0x01 to 0x06 or 0x08; the property already exists, so nothing moved). The ESP was
written via a temp name, sync, sha256 verify, and rename; the sealed file was
restored and read back after each arm; four M2 reboots, each back in about 95 s.
The live DT value was read from `/sys/firmware/devicetree` after each boot.

The overlay in this branch is the package form of the same one-property change
and has not been boot-tested through omarchy-mac-boot on a T6021 machine; the
unit test merges it into a minimal board tree and checks the value.

## Not measured

Burst energy at base 6 on this SoC; base 5 and 7; T6020 and T8112.

## Verification of this branch

`python3 -m unittest tests.test_gpu_pstate_t6021 tests.test_gpu_pstate_t6001
tests.test_gpu_pstate_overlay`: OK (6 tests).
