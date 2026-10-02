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
  documented in [omarchy-ane's probe guide](https://github.com/joshuawarren/omarchy-ane/blob/main/docs/ane-probe.md).
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

## Turn on the ANE for your chip

To submit a judged row for an untested chip, install omarchy-ane-dkms and add that chip's opt-in
key from the omarchy-ane README table to /etc/omarchy-platform/dtb-overlays.opt-in. For T6020,
T6022 and T8112, run sudo omarchy-ane-firmware-fetch first. Then run sudo omarchy-ane-dt apply
and reboot. From an omarchy-mlx checkout, run python3 scripts/collect_deep.py --ane-smoke
--submit. The collector runs the smoke when the chip is idle (load < 0.5, PSI 0); no fixed
uptime is required.

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
