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

## Kernel-side cost of base 6 versus stock (lab entry H291, `jwm1-parity/`, M1 only)

M1 (T8103). One boot per arm, a second stock boot to bound boot-to-boot drift,
idle-gated (load1 < 0.3, PSI 0), medians. Small operations that wait on the GPU
get much faster: the burst starts from the top state instead of ramping.
Memory-bound streams do not move: they already run back to back, at the top
state, in both arms.

| Operation (median us per op, control arm) | stock (base 1) | base 6 | macOS |
|---|---|---|---|
| add 2048, sync per op | 11.75 | 6.05 | — |
| rms_norm 2048, sync per op | 17.4 | 8.2 | — |
| silu*u 6144, sync per op | 15.6 | 9.9 | — |
| add 2048, pipelined | 5.12 | 4.54 | 2.00 |
| silu*u 6144, pipelined | 7.97 | 5.58 | 5.32 |

The macOS column is the same chain run in the macOS comparison window (H290,
`jwm1-parity/`); that window was busy, which biases macOS slower, so a macOS
lead in the table is conservative.

- qmv-class streaming bandwidth: unchanged. Every weight-read cell moved by
  0.7 % or less.
- GDN kernel-only times: unchanged (within 1 %).
- End-to-end decode and prefill: unchanged (45.8 to 46.0 tok/s in both arms,
  output digests exact).

Honest conclusion: base 6 helps small dependent operations and the first
request after a pause. It does not move memory-bound work. DVFS is not the qmv
gap. Do not turn it on to make decode or prefill faster; the gain there is
zero, and macOS keeps its lead on the pipelined cells.

## If DTBS is set, the opt-in does nothing

`update-m1n1` (omarchy-mac-boot) builds `boot.bin` from the board trees in
`/usr/lib/omarchy-mac-boot/dtb-overlays` and applies the overlays named in
`/etc/omarchy-mac-boot/dtb-overlays.opt-in`. A `DTBS=` setting in
`/etc/default/update-m1n1` replaces that whole mechanism: `update-m1n1` then
builds `boot.bin` from the files in that list only. No package overlay is
applied, and nothing prints a warning. An opt-in line plus a reboot changes
nothing, and `boot.bin` keeps its stock hash.

The aurora-sep kernel installer (Touch ID kernels) writes such a stanza:

```
# aurora-sep: build m1n1's stage 2 from the device trees the installed
# linux-aurora package owns, which carry the Touch ID sensor node.
DTBS=$(pacman -Qlq linux-aurora 2>/dev/null | grep '/dtbs/[^/]*\.dtb$'; true)
```

On one M1 install with this stanza (H291, attempt 1), the opt-in line and two
reboots left `apple,perf-base-pstate` at 1 and the stock `boot.bin` hash
unchanged. Check before you turn any opt-in overlay on:

```
scripts/check-dtbs-override.sh
```

The script reads the file only. Exit 1 and a WARNING mean package overlays are
ignored on this install. `python3 scripts/collect_deep.py` prints the same fact
in its steps.

### Temporary DTBS override (applies one overlay, then restores)

Keep the stanza: the Touch ID node and the protection against stale kernel
DTBs depend on it. Instead, override the list for one `update-m1n1` run.
Swap only your board tree for the overlay-merged copy; every other entry
stays the kernel's own. Keep a backup, and never leave the override in
place. The steps need `dtc`, `fdtoverlay` and `fdtget`; `<BOARD>` is your
board name (`t8103-j293` on a MacBookPro17,1), `<KVER>` the installed
`linux-aurora` version.

1. Build the merged tree and check the property landed. The package already
   ships the compiled overlay (inert while `DTBS` is set); build from a
   checkout only if the file is missing:

   ```
   dtbo=/usr/lib/omarchy-mac-boot/dtb-overlays/t8103/omarchy-gpu-pstate.dtbo
   fdtoverlay -i /usr/lib/modules/<KVER>/dtbs/<BOARD>.dtb \
       -o /tmp/<BOARD>-merged.dtb "$dtbo"
   fdtget -t u /tmp/<BOARD>-merged.dtb /soc/gpu@206400000 apple,perf-base-pstate
   ```

   The last command prints `6`. Without the package file:
   `dtc -@ -I dts -O dtb -o /tmp/gpu-pstate.dtbo packaging/dt/t8103-gpu-pstate.dts`
   from an mlx-omarchy checkout, and use that path as `$dtbo`.

2. Record what you fall back to:

   ```
   sudo cp /etc/default/update-m1n1 /etc/default/update-m1n1.bak
   sudo sha256sum /boot/efi/m1n1/boot.bin > /tmp/boot.bin.before.sha
   ```

3. Build the list: every `/dtbs/*.dtb` file from `pacman -Qlq linux-aurora`,
   with only `<BOARD>.dtb` swapped for the merged copy:

   ```
   pacman -Qlq linux-aurora | grep '/dtbs/[^/]*\.dtb$' \
       | sed "s|.*/<BOARD>\.dtb$|/tmp/<BOARD>-merged.dtb|" > /tmp/dtbs-with-merged.list
   grep -c . /tmp/dtbs-with-merged.list
   grep -c merged /tmp/dtbs-with-merged.list
   ```

   The first count is the full kernel DTB set; the second is 1. Append one
   line to `/etc/default/update-m1n1`:

   ```
   DTBS=$(tr '\n' ' ' < /tmp/dtbs-with-merged.list)
   ```

   `update-m1n1` runs with `set -e`, so the assignment must succeed; a list
   that fails to load stops the rebuild instead of building a broken
   `boot.bin`.

4. Rebuild, confirm the hash moved, reboot, and read the running tree:

   ```
   sudo update-m1n1 && sha256sum /boot/efi/m1n1/boot.bin
   od -An -tu4 --endian=big /sys/firmware/devicetree/base/soc/gpu@206400000/apple,perf-base-pstate
   ```

   The hash must differ from step 2, and `od` prints `        6`.

5. Restore in the same session. Remove the appended `DTBS=` line (or restore
   `update-m1n1.bak`), run `sudo update-m1n1`, and check `boot.bin` matches
   the step 2 hash again. Do not reboot with the temporary list still in the
   file after the merged copy is gone: a `boot.bin` that names a deleted file
   is a half-state.

Measured this way on one M1 (H291): the swapped list rebuilt `boot.bin`, the
next boot read `apple,perf-base-pstate` = 6, and the restore returned the
stock hash.

## Turn it on

First run `scripts/check-dtbs-override.sh`. Exit 1 means a `DTBS=` setting in
`/etc/default/update-m1n1` will ignore this opt-in; use the temporary-override
recipe in the section above instead.

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

## How users get GPU base pstate 6

Three ways exist. Today, only the documented manual override (c) works on an
install that carries a `DTBS=` stanza; the packaged opt-in works elsewhere.

(a) Make the boot hook honor the opt-in even with a DTBS stanza. Root cause,
from the omarchy-mac-boot sources: the packaged
`/etc/default/update-m1n1` calls `dtb_overlays_update_m1n1` as its last line,
and that call sets `DTBS` to the overlay-merged tree list. The aurora-sep
installer appends its own `DTBS=` line at the end of the file, so its
assignment runs after the call and the last write wins. The library also
returns silently when `DTBS` is already set. Direction adopted for the
boot-package lane: merge the overlays after the final assignment (for
example a RETURN trap that re-runs the merge once the whole file is read),
and teach the library to apply overlays to whatever `DTBS` list the config
left instead of bailing; keep the C-locale determinism. T8103 has the
measured data to ship this default-on (H245/H247: first token after a
100 ms-2 s pause about 9 ms faster, +26 mW at 1 Hz sparse load, none at
idle or saturation); T6001 and T6021 stay opt-in keys. Risk: the boot check
must reproduce the same merge, and a failed merge must keep the
"stays as the kernel shipped it" rule. Patch draft: the private receipt
`2026-10-05-overlay-hook/`.

(b) Change the default in the shipped device tree or the driver. Rejected:
it removes the opt-in, it would raise the base state on every chip at once
including T6021, whose base 6 is unmeasured, and a kernel or DT default has
no one-file rollback.

(c) The temporary DTBS override documented above. Manual, works today,
revertible in the same session.

Measured so far: the base-6 kernel-side and decode-side effects are M1
(T8103) numbers only (H291). T6001 measured no throughput change from any
base state. T6021 is unmeasured for base 6. Energy: only the 1 Hz sparse
cell is measured (+26 mW on T8103); battery runtime is unmeasured.

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
