# Help mlx-omarchy: share a capture from your machine

Want to help? The single most useful thing takes about ten minutes: run
our collector on your Apple-Silicon machine and submit the result. Every
capture fills in a row of the hardware matrix that decides which SoCs get
driver support next — see the
[chip-coverage table](https://github.com/joshuaswarren/omarchy-ane#chip-coverage).
If your machine's row says "data needed", a capture from you is literally
the unblock.

## What to do

The short version — clone, then submit with one command (`--submit`
already defaults to the public community endpoint):

```bash
python3 scripts/collect_quick.py --submit                        # ten seconds
python3 scripts/collect_deep.py --out mlx-omarchy-deep.tar.gz --submit
```

Everything you need — exact commands for Linux/Omarchy and for macOS,
what gets collected, and how redaction works — is in the
[README's **Contributing** section](../README.md#contributing), with more
detail in [CONTRIBUTING.md](../CONTRIBUTING.md).

The one-paragraph version:

- **No install, ten seconds (Linux/Omarchy):** the quick collector
  captures the ANE device-tree data and submits with one command —
  see [quick mode](../README.md#contributing).
- **Full report (either OS):** scripts/collect_deep.py adds the
  correctness sweep, benchmark numbers and ANE turn-on blocks. Linux
  includes reserved-memory, mailbox, firmware hashes, the omarchy_ane
  promotion block, and ane_linux.ane_probe: a read-only, privacy-redacted
  ANE topology and install diagnostic. Its 8 KiB output field list is
  documented in [omarchy-ane's probe guide](https://github.com/joshuaswarren/omarchy-ane/blob/main/docs/ane-probe.md).
  The server keeps it inline for querying. The report previews the exact
  redacted payload and sends nothing without --submit. An ANE smoke result
  can be added with --ane-smoke; it never loads or unloads modules and
  never writes. See [ane-turn-on-data.md](ane-turn-on-data.md) for coverage.
- **Dual-booters, you're gold:** run it under macOS *and* Omarchy on the
  same machine and submit both. The macOS side sees data (IORegistry,
  power topology) Linux can't, and vice versa — the pair is worth more
  than either alone.
- **A broken or partial capture can't hurt a submission:** the server
  sanitizes any diagnostics block it cannot parse into an `unparsed`
  field instead of rejecting the row, and the collector prints the
  server's full error body if a submit fails, so you see exactly why.
  The worker still accepts schema versions 1 and 2; the new probe field
  updates the schema identity without changing that version list.
- **PII handling, end to end (v0.7.23+):** the collector removes every
  serial / unique-id property from the dtc dump and the IORegistry dump
  before writing, and the server defensively scans every payload. If
  the scan still finds a leak (an old collector, a free-text note, a
  hostname mDNS suffix), the worker **strips** the matched value and
  stores the cleaned copy with `pii_redacted: {kind: n}` counts in the
  response. Nothing PII-shaped ever lands in the public row.

## Turn on the ANE for your chip

The collector prints the steps that match your kernel. One wording source:
`scripts/collect_deep.py`. Both variants, plus the one override:

- Kernel without the in-tree ANE driver: "To submit a judged row for an untested
  chip, install omarchy-ane-dkms and add that chip's opt-in key from the
  omarchy-ane README table to /etc/omarchy-mac-boot/dtb-overlays.opt-in. For
  T6020, T6022 and T8112, run sudo omarchy-ane-firmware-fetch first. Then run
  sudo omarchy-ane-dt apply and reboot. From an omarchy-mlx checkout, run
  python3 scripts/collect_deep.py --ane-smoke --submit. The collector runs the
  smoke when the chip is idle (load < 0.5, PSI 0); no fixed uptime is required."
- Kernel that ships the ANE driver in-tree: "To submit a judged row for an
  untested chip on a kernel that ships the ANE driver in-tree: userspace + smoke
  + firmware fetch only; do not install omarchy-ane-dkms. Add that chip's opt-in
  key from the omarchy-ane README table to
  /etc/omarchy-mac-boot/dtb-overlays.opt-in. For T6020, T6022 and T8112, run
  sudo omarchy-ane-firmware-fetch first. Then run sudo omarchy-ane-dt apply and
  reboot. From an omarchy-mlx checkout, run python3 scripts/collect_deep.py
  --ane-smoke --submit. The collector runs the smoke when the chip is idle
  (load < 0.5, PSI 0); no fixed uptime is required."
- "DTBS= is set in /etc/default/update-m1n1, so m1n1 boots the kernel's own
  device trees: the overlay opt-in has no effect, and the chip is enabled only
  by its node in the kernel DT." The collector appends that sentence when the
  DTBS= line there is not empty.

## Test the Neural Engine on a base M2 (T8112)

<a id="test-the-neural-engine-on-a-base-m2-t8112"></a>

Base M2 only: a MacBook Air 13" or 15", a 13" MacBook Pro, or a Mac mini with
the base M2 (T8112). Not M2 Pro, not M2 Max, not M2 Ultra. The opt-in key is
`ane-t8112`
([omarchy-ane chip table](https://github.com/joshuaswarren/omarchy-ane#chip-coverage)).

Run these six steps in order on stock Omarchy with current updates.

1. Update the system and install the driver package:

   ```sh
   sudo pacman -Syu && sudo pacman -S omarchy-ane-dkms
   ```
2. Add the opt-in key:

   ```sh
   echo ane-t8112 | sudo tee -a /etc/omarchy-mac-boot/dtb-overlays.opt-in
   ```
3. Fetch the ANE firmware. T8112 loads its own image, and the install hook
   covers only the M2 Max:

   ```sh
   sudo omarchy-ane-firmware-fetch
   ```
4. Apply the overlay, rebuild m1n1, reboot:

   ```sh
   sudo omarchy-ane-dt apply
   sudo update-m1n1
   sudo reboot
   ```
5. Check:

   ```sh
   omarchy-ane-check --smoke
   ```

   Expect the last line `omarchy-ane-check: ready` and a smoke line that ends
   `20/20 calls bit-exact`.
6. From a checkout of the latest omarchy-mlx release (v0.7.25 or newer), run:

   ```sh
   python3 scripts/collect_deep.py --ane-smoke --submit
   ```

   The collector runs the smoke when the chip is idle (load < 0.5, PSI 0);
   it waits up to 300 s and no fixed uptime is required. One passing row
   promotes T8112 from opt-in ([ane-turn-on-data.md](ane-turn-on-data.md)).

If `/etc/default/update-m1n1` has a non-empty `DTBS=` line (some custom
kernels), m1n1 boots the kernel's own device trees, the opt-in has no effect,
and `omarchy-ane-dt apply` refuses. The stock linux-aurora kernel has no ANE
node, so that setup is not supported for this test. The collector prints the
same note ([Turn on the ANE for your chip](#turn-on-the-ane-for-your-chip)).

If anything fails, open an issue on
[joshuaswarren/omarchy-ane](https://github.com/joshuaswarren/omarchy-ane/issues)
with the full `omarchy-ane-check --smoke` output.

## Timing note

Grab the **latest release** before running. If we've just announced a
release in progress, waiting a few hours for it is worth it — newer
releases collect more fields. Not sure? Run it anyway; captures from any
recent version are all useful, and the archive accepts them all.

## Your data

Redacted hostnames, paths, and serials; explicit preview; opt-in submit;
open [schema](../services/community-data/schema/) and [archive](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results).
Something failed or looks weird (a 422, a crash, an exotic machine)?
[Open an issue](https://github.com/joshuaswarren/omarchy-mlx/issues) and
paste the exact error text — the collector prints the failing field, and
that line is what we need.

A ten-minute capture from you can save days of bring-up work for a chip
family. Thank you.
