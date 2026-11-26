# GPU base performance-state overlays

GPU performance-state overlays are opt-in boot configuration changes.
For the M1, `gpu-pstate-t8103` sets `apple,perf-base-pstate` to `6` on
`/soc/gpu@206400000`. Check the boot configuration before enabling it, and
keep a rollback copy. Installing MLX does not require this overlay.

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

On one M1 install with this stanza (first attempt), the opt-in line and two
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
   from the repository root, and use that path as `$dtbo`.

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
   file after the merged copy is gone. Future `update-m1n1` runs still need
   every file named by `DTBS`; restore the configuration before deleting them.

Measured this way on one M1: the swapped list rebuilt `boot.bin`, the
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

