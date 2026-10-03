# GPU base performance state on T8103 (opt-in overlay)

`gpu-pstate-t8103` raises the Apple GPU firmware's base performance state on
the M1 from 1 (396 MHz) to 6 (1278 MHz). It is an **opt-in** device tree
overlay. It is off unless the owner turns it on.

## What it does

The GPU firmware (AGX) starts every burst from its base state and ramps up. The
GPU powers itself off about 2 ms after the last command. A request that arrives
100 ms to 2 s after the previous one therefore pays the ramp from 396 MHz. A
back-to-back run (decode, a long prefill) does not: the firmware is already at
the top state.

The overlay sets one property on one node, `apple,perf-base-pstate = <6>` on
`/soc/gpu@206400000`. The firmware reads it at GPU init, so it takes effect at
the next boot, after `update-m1n1` has rebuilt `boot.bin`.

## Measured (MacBookPro17,1, kernel 7.1.12-2-7-ARCH, Qwen3.8-2B)

Lab entries H236, H237, H245, H247 in `jwm1-parity/`. Idle-gated: load1 < 0.5,
PSI 0, 7+ minutes after boot. Medians.

| Request after an idle gap of | stock (base 1) | base 3 | base 6 |
|---|---|---|---|
| 20 ms | 109.0 ms | 109.0 | 110.0 |
| 100 ms | 118.1 ms | 115.8 | 110.4 |
| 500 ms | 119.0 ms | 114.8 | 110.3 |
| 2000 ms | 126.0 ms | 121.6 | 115.4 |

Base 6 removes about 95 % of the post-pause penalty at 100 and 500 ms and about
60 % at 2 s. Base 3 removes about 35 to 40 %.

| Whole-machine power (SMC Total System Power) | stock | base 6 |
|---|---|---|
| idle, 180 s median | 3536 mW | 3529 mW |
| 1 request/s for 100 s, mean | 7615 / 7634 / 7611 mW (three stock runs) | 7649 mW |
| saturated, always busy, mean | 17306 to 17390 mW | 17397 mW |

At one request per second base 6 draws 26 mW (0.34 %) more than the mean of the
stock runs, which is inside the 24 mW spread between stock runs. The firmware
holds the high state only while the GPU is on. The rail is not separately
metered; this is the whole machine.

Back-to-back throughput is unchanged: decode 45.0 to 45.2 tok/s and prefill
about 414 tok/s in every arm, with the output digests exact. **This lever does
not move the macOS comparison cells**, which run back to back.

## Turn it on

```
echo gpu-pstate-t8103 | sudo tee -a /etc/omarchy-platform/dtb-overlays.opt-in
```

then let the overlay consumer rebuild the boot device tree and reboot
(`omarchy-ane-dt apply` on a system with `omarchy-ane-dkms`, or the newer
`omarchy-mac-boot`, which applies `/usr/share/omarchy-platform/dtb-overlays`
itself; both read the opt-in file). Check after the reboot:

```
od -An -tu4 --endian=big /sys/firmware/devicetree/base/soc/gpu@206400000/apple,perf-base-pstate
```

prints `6`. To turn it off, remove the line from the opt-in file and rebuild.

## Limits

- Only T8103 has been measured. T6001 (M1 Max) and T6021 (M2 Max) have other
  OPP tables and node names. Each chip gets its own overlay after its own A/B.
  The default stays stock until those exist.
- The effect is a latency gain after a pause. Nobody has measured the energy of
  short bursts at a high base state on battery separately from whole-machine
  power.
- Other properties of the same node (`apple,perf-tgt-utilization`,
  filter and gain constants) are not changed. They are the subject of H248.

## Package

`packaging/dt/t8103-gpu-pstate.dts` is the source. `packaging/build-dtbo.sh
"$pkgdir"` compiles it to
`usr/share/omarchy-platform/dtb-overlays/t8103/omarchy-gpu-pstate.dtbo`. The
recipe needs `dtc` at build time. `tests/test_gpu_pstate_overlay.py` builds the
overlay and merges it into a minimal board tree.
