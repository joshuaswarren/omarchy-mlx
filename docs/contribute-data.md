# Help mlx-omarchy: share a capture from your machine

Want to help? The single most useful thing takes about ten minutes: run
our collector on your Apple-Silicon machine and submit the result. Every
capture fills in a row of the hardware matrix that decides which SoCs get
driver support next — see the
[chip-coverage table](https://github.com/joshuaswarren/omarchy-ane#chip-coverage).
If your machine's row says "data needed", a capture from you is literally
the unblock.

## What to do

Everything you need — exact commands for Linux/Omarchy and for macOS,
what gets collected, and how redaction works — is in the
[README's **Contributing** section](../README.md#contributing), with more
detail in [CONTRIBUTING.md](../CONTRIBUTING.md).

The one-paragraph version:

- **No install, ten seconds (Linux/Omarchy):** the quick collector
  captures the ANE device-tree data and submits with one command —
  see [quick mode](../README.md#contributing).
- **Full report (either OS):** `scripts/collect_deep.py` adds the
  correctness sweep and benchmark numbers; it shows you the exact
  redacted payload and sends nothing without your explicit `--submit`.
- **Dual-booters, you're gold:** run it under macOS *and* Omarchy on the
  same machine and submit both. The macOS side sees data (IORegistry,
  power topology) Linux can't, and vice versa — the pair is worth more
  than either alone.

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
