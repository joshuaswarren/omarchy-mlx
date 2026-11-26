# Boot and kernel on Apple Silicon Linux

This is the reference for how the kernel boots and how the device tree is
built on the M1 test machine. It also lists the requirements for the Omarchy
Linux kernel for Mac.

Facts were verified on one Apple M1 (`t8103`) running Omarchy with kernel
`7.1.6-1-1-ARCH`. Items marked **Hypothesis** are not proven.

## How the device tree reaches the kernel

The boot chain is iBoot, m1n1, U-Boot, GRUB, kernel.

The kernel never reads a `.dtb` out of `/boot`. m1n1 carries the device
trees inside its own payload and patches the matching board `.dtb` on every
boot.

`/usr/bin/update-m1n1` builds that payload. It picks the device trees and
concatenates m1n1, the `.dtb` files, gzipped U-Boot, and the m1n1 config:

```sh
: ${DTBS:=$(/bin/ls -d /lib/modules/*-ARCH | sort -rV | head -1)/dtbs/*.dtb}
```

```sh
cat "$M1N1" $DTBS >"${TARGET}.new"
gzip -c "$U_BOOT" >>"${TARGET}.new"
cat "$m1n1config" >>"${TARGET}.new"
```

`TARGET` defaults to the system ESP path `m1n1/boot.bin`. The ESP is mounted
at `/boot/efi` on the test machine, so the payload is
`/boot/efi/m1n1/boot.bin`. When the new payload differs from the current
one, update-m1n1 keeps the previous payload as `boot.bin.old`.

A pacman hook re-runs update-m1n1 after every kernel or boot-component
upgrade. `/usr/share/libalpm/hooks/95-m1n1-install.hook`:

```ini
[Trigger]
Type = Path
Operation = Install
Operation = Upgrade
Target = usr/lib/modules/*/dtbs/*
Target = usr/lib/asahi-boot/*

[Action]
Description = Updating m1n1 image...
When = PostTransaction
Exec = /usr/bin/update-m1n1
NeedsTargets
```

m1n1 patches the base `.dtb` on every boot and injects live values:

- `cpu-release-addr` for each CPU
- `kaslr-seed`
- the `framebuffer` node
- `asahi,m1n1-stage1-version` and `asahi,m1n1-stage2-version`
- `asahi,iboot1-version` and `asahi,iboot2-version`
- `asahi,system-fw-version` and `asahi,os-fw-version`
- `asahi,efi-system-partition`
- `linux,uefi-mmap-*`
- `bootargs`

A `.dtb` sitting in `/boot` is therefore not what the kernel gets. The
kernel gets the m1n1-patched tree from the payload. A bootloader command can
override that tree; that override is the subject of the next section.

## Why replacing the whole tree is wrong

GRUB's `devicetree` command replaces the complete m1n1-patched tree with
one static file. Every injected value above is lost.

Those values change between boots. The frozen snapshot in `/boot/ane.dtb`
showed the drift. Its `cpu@1` `cpu-release-addr` is `0x8_05b14230`, while
the live boot placed it at `0x8_04f9c230`. The snapshot also froze a stale
`kaslr-seed`, `framebuffer@bd5d70000`, `asahi,m1n1-stage2-version
v1.5.2`, and `asahi,iboot2-version iBoot-8422.141.2`.

Regenerating the snapshot once per boot does not fix this. The per-CPU
release addresses moved between the two observed boots, so a fresh snapshot
is stale as soon as the next boot starts.

Under the override boot the kernel failed to bring up the secondary CPUs.
The log showed `smp: Bringing up secondary CPUs ...`, then `failed to come
online` for CPU1 through CPU7, each with `failed in unknown state : 0x0`.

**Hypothesis:** the frozen per-CPU `cpu-release-addr` is the leading
explanation. Each secondary CPU spins at the address from the old boot,
while the live boot placed the release stub elsewhere. The A/B test below
proves that the boot entry decides the outcome. It does not separate this
explanation from the other frozen values; the snapshot also freezes the
memory map and the reserved regions. Treat the mechanism as unproven.

## The one-core incident

Symptom. `nproc` printed `1`. `/sys/devices/system/cpu/online` printed
`0` and `offline` printed `1-7`. The per-CPU `online` files did not exist,
so `chcpu` printed `CPU 1 is not hot pluggable`.

The kernel was not the cause. `CONFIG_SMP=y`, `CONFIG_NR_CPUS=64`, and
`CONFIG_HOTPLUG_CPU=y` were all set. The device tree described all 8 CPUs
with enable-method `spin-table`.

How it was found. `/boot/efi/grub-ane/grub.cfg` had `set default=1`. That
entry ran `devicetree /@/boot/ane.dtb` and so replaced the live tree with
the frozen snapshot.

The A/B proof. Booting entry 1 gave 1 core. Booting entry 0 gave 8 cores,
with `online` printing `0-7`. This proves the boot entry was responsible.
It does not prove which frozen value caused the failure.

Fix applied. `default=0` restored the stock boot. The test entry was renamed
to `Omarchy ANE test (STALE DT: boots 1 core only)` so nobody selects it by
accident. The previous grub config is kept at
`/boot/efi/grub-ane/grub.cfg.bak-20260903-130516`.

Blast radius on published timings. GPU-bound results barely moved: the
matmul median went from `0.1556` to `0.1576` TFLOP/s across the fix.
Host-bound tokens/s figures measured during the one-core window are the
suspect ones. Re-measure them before further publication.

## ANE on Linux today

Status: hardware present, boot path missing, driver dangerous.

- The hardware is present and the kernel binds a platform device:
  `platform 26bc04000.ane: Adding to iommu group 2`.
- The `ane@26a000000` node exists only under the override boot, because it
  comes from the frozen `/boot/ane.dtb`. The packaged
  `t8103-j293.dtb` does not carry it:
  `strings /lib/modules/7.1.6-1-1-ARCH/dtbs/t8103-j293.dtb | grep -c
  t8103-ane` prints `0`.
- The node shape:

  ```dts
  compatible = "apple,t8103-ane";
  reg-names = "engine", "dart0", "dart1", "dart2";
  interrupt-names = "ane dart";
  ```

  `reg` has four ranges, `interrupts` are `0x1a0` and `0x1a1`, and the node
  also has `power-domains` and `iommus`.

- Open decision to settle before anything ships: the node crams three DARTs
  into one `reg` block. That matches the out-of-tree ANE driver binding
  (`eiln/ane` lineage, carried today by `joshuaswarren/omarchy-ane`), not
  the mainline shape of separate DART nodes with `iommus` phandles. The
  binding must be picked before the in-tree node is written.
- The driver builds. `~/src/apple-ane-kmd/ane/ane.ko` has a `vermagic`
  that matches the running kernel exactly. Loading it hard-reset the
  machine: ssh died and the box reset itself, which wiped `/tmp`. Do not
  load it again on a host doing other work.
- `CONFIG_DRM_ACCEL=y` and `CONFIG_OF_OVERLAY=y` are already set. Mainline
  has no userspace configfs DT-overlay loader, so an overlay must come from
  the bootloader or the node must be in-tree. U-Boot in
  `/usr/lib/asahi-boot/u-boot-nodtb.bin` does carry fdt support (`fdt
  resize`, `fdtoverlays` strings), but the payload selection still goes
  through m1n1 first.

## Omarchy kernel requirements

Checklist for the packager of the Omarchy Linux kernel for Mac.

- [ ] Carry the `ane@26a000000` node in the in-tree DTS. Set
      `status = "disabled"` by default and enable it per board file.
      Mainline has no userspace overlay loader, so the node must ship in
      the tree the bootloader hands to the kernel.
- [ ] Install the compiled `.dtb` files under
      `/usr/lib/modules/<kernel-version>/dtbs/`. That is the only place
      update-m1n1 reads.
- [ ] Name the modules directory to match the `*-ARCH` glob, or ship
      `/etc/default/update-m1n1` with `DTBS` set. update-m1n1 sources that
      file before the default is applied, so `DTBS` overrides the glob.
      No such file exists on the test machine today.
- [ ] Know the single-directory trap. `sort -rV | head -1` picks exactly
      one `*-ARCH` directory, the highest version. A kernel installed as
      `/lib/modules/7.1.6-1-omarchy` contributes nothing: with no `*-ARCH`
      directory left, update-m1n1 exits with
      `ERROR: DTBS config unset or empty`; with an older `*-ARCH` kernel
      still installed, the payload silently keeps that older kernel's
      `.dtb` files and the new kernel boots with foreign device trees.
- [ ] Handle multiple installed kernels. Only the newest `*-ARCH`
      directory reaches the payload. The pacman hook re-runs update-m1n1
      on every install and upgrade, but the packager must not leave a
      `*-ARCH` directory that is newer than the intended kernel.
- [ ] Ship the DTS and the ANE driver from the same source version, in one
      package set. A driver built out of tree against a different kernel
      release breaks on `vermagic`; today's match was a same-version
      build, not a guarantee.
- [ ] Keep the driver blacklisted until it stops resetting the machine.
      One load hard-reset the M1.
- [ ] Keep the verified config values: `CONFIG_SMP=y`,
      `CONFIG_NR_CPUS=64`, `CONFIG_HOTPLUG_CPU=y`, `CONFIG_DRM_ACCEL=y`,
      `CONFIG_OF_OVERLAY=y`.

## Verification checklist

Run these in order after changing a `.dtb`. Each step proves one link in
the chain: packaged tree, payload, then booted kernel.

1. The packaged tree carries the node:

   ```sh
   strings "/lib/modules/$(uname -r)/dtbs/t8103-j293.dtb" | grep -c t8103-ane
   ```

   The count is `0` today. It must be `1` or more before any boot test.

2. The hook rebuilt the payload. Hash the payload before and after the
   pacman transaction, and look for the backup:

   ```sh
   sha512sum /boot/efi/m1n1/boot.bin
   ls -la /boot/efi/m1n1/
   ```

   `boot.bin.old` appears only when the content changed.

3. After a reboot, confirm the node reached the kernel and the CPUs came
   up:

   ```sh
   nproc
   cat /sys/devices/system/cpu/online
   find /proc/device-tree -name '*ane*'
   cat /proc/device-tree/ane@26a000000/compatible
   ```

   Require `nproc` to print `8`, `online` to print `0-7`, and
   `compatible` to print `apple,t8103-ane`.

4. Confirm the tree is the live m1n1 tree, not a frozen copy:

   ```sh
   find /proc/device-tree -name 'asahi,*' | sort
   ```

   The m1n1-injected properties must be present.

## The decisive test still open

The one test that settles the ANE boot path:

1. Add the `ane@26a000000` node to the packaged `t8103-j293.dtb` or to the
   in-tree DTS.
2. Run `/usr/bin/update-m1n1`.
3. Boot the stock entry (`default=0`), not the stale-DT entry.
4. Require both results: `nproc` prints `8`, and the ANE node is present
   in `/proc/device-tree`.

Success proves the node boots with all eight cores. Failure localizes the
problem to the node itself.

Safety rails that exist:

- update-m1n1 keeps the previous payload as `boot.bin.old`.
- The grub config implements a one-shot guard. It reads `ane_trying` and
  `next_entry` from the grub environment and falls back to entry 0:

  ```sh
  if [ "${ane_trying}" = "1" ] ; then
     set default=0
     set ane_trying=
     save_env ane_trying
  fi
  if [ "${next_entry}" ] ; then
     set default="${next_entry}"
     set next_entry=
     save_env next_entry
  fi
  ```

- `grub-reboot` and `grub-set-default` are installed and drive this guard.

The test needs owner approval first. A bad `.dtb` in the payload can leave
the machine unbootable, and recovery is physical. Run it only on a machine
that does no other work.

## What the community collectors report

The mlx-omarchy community collectors (`collect_quick.py`,
`collect_deep.py`) submit to
`https://mlx-omarchy-community-data.joshua-s-warren.workers.dev`. The
payload schema is
`services/community-data/schema/payload-v1.schema.json`; the worker
rejects unknown fields.

The payload carries `cpu_online`, `kernel`, `chip`, and a `benchmark`
array with `tflops` and `median_ms` per case.

This failure has a signature in that data: `cpu_online` is `1` while the
GPU-bound `tflops` values look normal. A one-core machine produces
plausible GPU numbers and unusable host-bound numbers. On the test
machine that signature came from a bootloader device-tree override, so
treat a low `cpu_online` on a `t8103` submission as the first suspect.
