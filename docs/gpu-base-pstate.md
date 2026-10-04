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
echo gpu-pstate-t8103 | sudo tee -a /etc/omarchy-mac-boot/dtb-overlays.opt-in
```

then run `sudo update-m1n1` (omarchy-mac-boot builds `boot.bin` from the
overlaid board trees in `/usr/lib/omarchy-mac-boot/dtb-overlays` and reads the
opt-in file) and reboot. Check after the reboot:

```
od -An -tu4 --endian=big /sys/firmware/devicetree/base/soc/gpu@206400000/apple,perf-base-pstate
```

prints `6`. To turn it off, remove the line from the opt-in file and rebuild.

## T6001 (M1 Max)

A second overlay, opt-in key `gpu-pstate-t6001`, sets base state 6 on
`/soc/gpu@406400000`. The A/B on an M1 Max (greedy decode and prefill, Qwen3.8-2B, one boot per arm, stock bookends to bound
boot-to-boot drift) found no throughput change: decode at 512 tokens moved by
-0.32% to +0.08% across base 6, base 4 and target-utilization 40 and 60, and
prefill stayed within 0.2%. Greedy digests matched in every arm. The time to
first token after a pause fell with base 6 from 88.6, 88.6 and 91.7 ms to 84.4,
83.9 and 86.6 ms at gaps of 100, 500 and 2000 ms (about 5%). Base 4 gave about
half of that. Target-utilization 40 and 60 changed nothing. The chip stays on stock by default.
Details: `docs/gpu-base-pstate-t6001.md`.

## Limits

- T8103 and T6001 have been measured. T6021 (M2 Max) has another OPP table and
  node name and gets its own overlay after its own A/B. The default stays
  stock.
- The effect is a latency gain after a pause. Nobody has measured the energy of
  short bursts at a high base state on battery separately from whole-machine
  power.
- Other properties of the same node (`apple,perf-tgt-utilization`,
  filter and gain constants) are not changed by this overlay.
- Packaged path verified on one M1 (H248): the built overlay, the opt-in line,
  `update-m1n1` and a reboot gave `apple,perf-base-pstate` = 6 with the
  target-utilization value and the ANE node unchanged. Removing the files,
  running `update-m1n1` and rebooting restored base state 1 and the stock
  `boot.bin` hash. The ANE smoke did not run in that boot (it is not in the
  0.4.0 package). A separate lab-overlay boot with base state 6 passed the ANE
  smoke 20 of 20 (H246).
- Other base states on the same chip (H248): base 4 removes 62-65% of the
  post-pause latency penalty, base 3 removes 38%, and target-utilization 40
  removes 30% and moves no sustained cell. Base 6 is the shipped value.

## Package

`packaging/dt/t8103-gpu-pstate.dts` is the source. `packaging/build-dtbo.sh
"$pkgdir"` compiles it to
`usr/lib/omarchy-mac-boot/dtb-overlays/t8103/omarchy-gpu-pstate.dtbo`. The
recipe needs `dtc` at build time. `tests/test_gpu_pstate_overlay.py` builds the
overlay and merges it into a minimal board tree.
