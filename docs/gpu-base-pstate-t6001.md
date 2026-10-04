# GPU base performance state on T6001 (opt-in overlay)

`gpu-pstate-t6001` raises the Apple GPU firmware's base performance state on
the M1 Max from 1 (388.8 MHz) to 6 (1296 MHz, the top of the
off/388.8/486/648/777.6/972/1296 MHz OPP table). It is an **opt-in** device
tree overlay. It is off unless the owner turns it on.

## What it does

Same mechanism as the T8103 overlay (`docs/gpu-base-pstate.md`): the AGX
firmware starts every burst from its base state and ramps up; a request that
arrives after an idle pause pays the ramp. The overlay sets one property on
one node, `apple,perf-base-pstate = <6>` on `/soc/gpu@406400000`. The kernel
driver (`drivers/gpu/drm/asahi/hw/mod.rs`) parses it for every SoC and passes
it to the firmware, which reads it at GPU init — effective at the next boot,
after `update-m1n1` has rebuilt boot.bin.

## Measured (M1 Max, MacBookPro18,2-class, kernel 7.1.13-3-2-ARCH, Qwen3.8-2B-mlx-4Bit)

Lab entry `Jw16Dvfs/20261004T002600Z-jw16-gpu-dvfs-dt-ab.md`; repo receipt
`receipts/2026-10-04-jw16-gpu-dvfs/`. Idle-gated (load1 < 0.5, PSI 0, 7.5+
min after boot), one boot per arm, stock re-measured on both ends.

| Request after an idle gap of | stock (base 1) | base 6 |
|---|---|---|
| 20 ms | 85.4-88.7 ms | 86.0 ms |
| 100 ms | 88.6 ms | 84.4 ms |
| 500 ms | 88.6 ms | 83.9 ms |
| 2000 ms | 91.7-92.6 ms | 86.6 ms |

Base 6 removes the post-pause penalty (4-5 ms, -5 % of a ~86 ms first token)
at 100 ms-2 s gaps. Base 4 (777.6 MHz) recovers roughly half. Back-to-back
throughput does not move in ANY arm — decode d64..d512 102.5-109.6 tok/s and
prefill 1377-1401 tok/s within +-0.5 % of stock (inside boot-to-boot drift),
output digests bit-exact everywhere. `apple,perf-tgt-utilization` 40/60
(alternative arms) changed nothing, cells or gaps: the governor target does
not change the clock a burst starts from.

Whole-machine power was NOT measured on this host (no SMC sampler installed);
the energy side is carried by T8103 H247: no idle cost, +26 mW (0.34 %) at
1 request/s bursts, none at saturation.

## Turn it on

On a system with `omarchy-mac-boot` (which consumes `/usr/lib/omarchy-mac-boot/dtb-overlays`
plus `/etc/omarchy-mac-boot/dtb-overlays.opt-in`):

```
echo gpu-pstate-t6001 | sudo tee -a /etc/omarchy-mac-boot/dtb-overlays.opt-in
```

then run `sudo update-m1n1` (omarchy-mac-boot builds `boot.bin` from the
overlaid board trees) and reboot. Check after the reboot:

```
od -An -tu4 --endian=big /sys/firmware/devicetree/base/soc/gpu@406400000/apple,perf-base-pstate
```

prints `        6`. To turn it off, remove the line and rebuild.

## Limits

- Only T6001 has been measured on this axis's throughput side; T6021 needs
  its own overlay and A/B. The default stays stock until the owner opts in.
- The gain is interactive (first token after a pause), not the parity ledger.
- jw16's installed boot path has no omarchy-mac-boot; there the overlay is
  applied by `omarchy-ane-dt`'s board-copy mechanism (measured that way).

## Package

`packaging/dt/t6001-gpu-pstate.dts` is the source. `packaging/build-dtbo.sh
"$pkgdir"` compiles it to
`usr/lib/omarchy-mac-boot/dtb-overlays/t6001/omarchy-gpu-pstate.dtbo`.
`tests/test_gpu_pstate_t6001.py` builds the overlay and merges it into a
minimal board tree.
