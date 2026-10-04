# GPU base performance state on T6021 (opt-in overlay)

`gpu-pstate-t6021` raises the Apple GPU firmware's base performance state on
the M2 Max from 1 (444 MHz) to **6 (1236 MHz)** of the
off/444/612/808/968/1110/1236/1338/1398 MHz OPP table. It is an **opt-in**
device tree overlay. It is off unless the owner turns it on.

**6 is not the top state (8, 1398 MHz), on purpose.** Base 8 was measured and is
the wrong value on this SoC (below). On T8103 and T6001 the value that works is
the top state; do not copy that rule to the M2 Max.

## What it does

Same mechanism as the T8103 (`docs/gpu-base-pstate.md`) and T6001
(`docs/gpu-base-pstate-t6001.md`) overlays: the AGX firmware starts every burst
from its base state and ramps up; a request that arrives after an idle pause pays
the ramp. The overlay sets one property on one node, `apple,perf-base-pstate = <6>`
on `/soc/gpu@406400000`. The firmware reads it at GPU init, so it takes effect at
the next boot, after `update-m1n1` has rebuilt `boot.bin`.

## Measured (MacBook Pro 14" M2 Max, j414c, kernel 7.1.13-3-1-ARCH, Qwen3.8-2B-mlx-4Bit, v0.7.24)

Lab entry `jwm1-parity/20261004T033000Z-m2-gpu-base-pstate-offline-build-h253.md`
(H253). Idle-gated: load1 <= 0.46, PSI 0, every arm measured 7.5 min or more after
boot; one boot per arm; stock measured on both ends (S0 at 1.7 h uptime, S1 at
7.5 min: they agree within 0.3 %, so boot age does not explain any difference).
The overlay was applied to the board device tree inside the boot image
(one byte of the j414c DTB changed); the live value was read back from
`/sys/firmware/devicetree` after each boot. Output digests are identical in
every arm.

Median first-token latency after an idle gap (ms, n = 24 per gap):

| gap | stock S0 | stock S1 | base 8 (1398) | base 6 (1236) |
|---|---|---|---|---|
| 20 ms | 85.3 | 83.9 | 87.2 | 84.4 |
| 100 ms | 91.0 | 87.2 | 87.2 | 84.9 |
| 500 ms | 89.5 | 88.3 | 87.4 | 84.7 |
| 2000 ms | 94.3 | 93.0 | 91.1 | 87.6 |

Base 6 removes the post-pause penalty at 100 ms and 500 ms and about two thirds
of it at 2 s: 4.2 / 4.2 / 6.1 ms (4.7 / 4.7 / 6.5 %) faster than the mean of the
two stock arms, with no loss on back-to-back requests or at short gaps.

Back-to-back cells (mean of 2 reps):

| cell | stock S0 / S1 | base 8 | base 6 |
|---|---|---|---|
| decode d64 / d128 / d256, tok/s | 109.6 / 109.6 / 108.5 (S0), 109.9 / 109.5 / 108.6 (S1) | 107.6 / 107.5 / 106.6 (-1.9 %) | 110.0 / 109.6 / 108.5 (0.0 %) |
| prefill 512, tok/s | 1608 / 1610 | 1557 (-3.3 %) | 1611 (+0.1 %) |
| first token, 512-token prompt, s | 0.0906 / 0.0900 | 0.0925 (+2.3 %) | 0.0906 |

Idle power (180 s median, SMC Total System Power): stock 14.92 / 15.03 W, base 8
15.20 W, base 6 15.12 W. The stock-to-stock spread is 0.1 W, so base 6 is about
one percent over stock and inside the noise. Base 8 draws more, loses throughput,
and is slower for a request that comes 20 ms after the previous one. It does not
belong in an opt-in that is meant to be safe to turn on.

Burst energy at a high base state was NOT measured for base 6 on this SoC (the
base 8 arm: 20.65 W at one request per second against 20.59 / 20.66 W stock).
Base 7 and base 5 were not measured.

## Turn it on

On a system with `omarchy-mac-boot` (which consumes
`/usr/lib/omarchy-mac-boot/dtb-overlays` plus
`/etc/omarchy-mac-boot/dtb-overlays.opt-in`):

```
echo gpu-pstate-t6021 | sudo tee -a /etc/omarchy-mac-boot/dtb-overlays.opt-in
```

then run `sudo update-m1n1` and reboot. Check after the reboot:

```
od -An -tu4 --endian=big /sys/firmware/devicetree/base/soc/gpu@406400000/apple,perf-base-pstate
```

prints `        6`. To turn it off, remove the line and rebuild.

## Limits

- Only T6021 (M2 Max) has been measured. T6020 (M2 Pro) and T8112 (M2) have
  other OPP tables; each needs its own overlay and A/B. The default stays stock
  until the owner opts in.
- The gain is interactive (first token after a pause), not a throughput gain.
- The value is per SoC. Reusing the T8103 or T6001 overlay's "top state" idea
  here would pick 8 and lose 2-3 % of sustained throughput.

## Package

`packaging/dt/t6021-gpu-pstate.dts` is the source. `packaging/build-dtbo.sh
"$pkgdir"` compiles it to
`usr/lib/omarchy-mac-boot/dtb-overlays/t6021/omarchy-gpu-pstate.dtbo`.
`tests/test_gpu_pstate_t6021.py` builds the overlay, merges it into a minimal
board tree, and checks the value is 6.
