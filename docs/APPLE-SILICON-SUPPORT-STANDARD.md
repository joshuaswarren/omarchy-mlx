# Apple Silicon support standard

This is the release-certification contract for every Omarchy-MLX release tag. The GPU/ANE compute-status table in the README remains separate. Platform repositories that adopt this standard should link the same certification record from their release.

## Status tiers

- **Certified:** the SoC/board is in the release-certification set and has a named hardware owner. Publishing the claim for a release still requires its green update gate and exact-candidate owner sign-off.
- **Supported:** this project claims the platform ships and is expected to work on the SoC/board, but it is not in the release-certification set.
- **Experimental:** useful bring-up exists, but it is outside the supported release.
- **Not yet:** no release support claim.

## Current matrix

| Generation | SoC / board | Tier | Release owner | Basis |
|---|---|---|---|---|
| M1 | T6000 / `J314s` (MacBookPro18,3), exact board | Certified | Sean Dolbec ([@techmore](https://github.com/techmore)) | The platform's [M1 Pro qualification host](https://github.com/omacom/omarchy-mac-pkgs/blob/28abebee14de05c35a2caa8dd257d056d15a641e/mac-manual/content/12-hardware.md) and full-regression machine. |
| M1 | T8103; T6000 other boards; T6001; T6002 | Supported | **Wanted** | M1 is in the [shipped platform family](https://github.com/omacom/omarchy-mac-pkgs/blob/28abebee14de05c35a2caa8dd257d056d15a641e/mac-manual/content/12-hardware.md); this repository separately records [T8103 and T6001 compute runs](compatibility.md). These are not the named platform release host. |
| M2 | T8112; T6020 / `J414s`; T6021 / `J416c`; T6022 | Supported | **Wanted** | M2 is in the [shipped platform family](https://github.com/omacom/omarchy-mac-pkgs/blob/28abebee14de05c35a2caa8dd257d056d15a641e/mac-manual/content/12-hardware.md). The platform calls T6021 a reference host, but this standard has no accountable M2 owner; an owner is needed for promotion to Certified. |
| M3 | T8122; T6030; T6031 / `J514c` | Experimental | **Wanted** | T8122 and T6030 [display/GPU handoff is available to edge testers](https://github.com/omacom/omarchy-mac-pkgs/blob/28abebee14de05c35a2caa8dd257d056d15a641e/mac-manual/content/12-hardware.md); T6031 remains bring-up-only. Compute is not certified. |
| M4 | T8132 / `J773g` Mac mini, exact board | Experimental | Sean Dolbec ([@techmore](https://github.com/techmore)) | A [J773g GPU run exists](m4-linux.md) with an out-of-tree driver stack; it is not a supported platform release yet. |
| M4 | T6040 / `J773s` Mac mini Pro and `J614s` / `J616s` MacBook Pro | Not yet | **Wanted** | [Community board identities are known](https://github.com/AsahiLinux/docs/blob/ed6f9d6ac4ac49a09924224ac07e7341a71e1e45/docs/hw/devices/device-list.md), but the project has [no qualifying T6040 run](m4-linux.md). |

Unless a row says “exact board,” its tier covers every board using the listed SoC; board codes in that row are examples, not an exhaustive list. Any unlisted SoC, and any sibling board of an exact-board row not covered elsewhere, is **Not yet**. This includes T6041.

Sources: the platform [hardware manual](https://github.com/omacom/omarchy-mac-pkgs/blob/28abebee14de05c35a2caa8dd257d056d15a641e/mac-manual/content/12-hardware.md), this repository's [compute table](../README.md#mlx-compute-coverage), [tested-hardware records](TESTED-HARDWARE.md), the [M4 bring-up record](m4-linux.md), and the community [Apple device list](https://github.com/AsahiLinux/docs/blob/ed6f9d6ac4ac49a09924224ac07e7341a71e1e45/docs/hw/devices/device-list.md).

## Release checklist

`CI` is automatic. For Certified hardware, `HW` is a dated sign-off from the matrix owner on the exact release candidate. Sean is currently the sole Certified owner; if he is unavailable, releases remain blocked until a named delegate signs on the same board. For Supported hardware without an owner, the release record must name the maintainer or community tester who ran each affected check; until someone is available, those checks are unverified rather than silently passed. Experimental machines publish results but do not block a release. **Not yet** machines make no release claim.

Keep the evidence in `docs/certifications/<tag>.md`, with the tested candidate SHA, linked CI job, tester, date and result for every required row. The release maintainer must also list the affected change categories, based on the candidate's included pull requests; that declaration determines which conditional Supported checks apply. After testing, one docs-only commit may add the certification file; tag the docs commit, and certify the unchanged candidate SHA named inside it.

| Check | Certified | Supported | Experimental | Automatic coverage |
|---|---|---|---|---|
| Boot and bootloader | HW every release | HW for boot changes | Record result | — |
| Internal display | HW every release | HW for display changes | Record result | — |
| HDMI / USB-C display | HW every release | HW for display changes | Record result | — |
| GPU acceleration | HW every release | HW for GPU changes | Record result | — |
| USB | HW every release | HW for kernel changes | Record result | — |
| NVMe | HW every release | HW for kernel changes | Record result | — |
| Wi-Fi / Ethernet | HW every release | HW for network changes | Record result | — |
| Audio | HW every release | HW for audio changes | Record result | — |
| Suspend / resume | HW every release | HW for kernel changes | Record result | — |
| Update from previous release | CI + HW every release | CI; HW for updater changes | Record result | Package resolvability plus real upgrade simulation |
| Rollback / recovery | HW every release | HW for updater/boot changes | Record result | Snapshot/rollback simulation where available |

The update gate is an external prerequisite whose workflow and successful job must be linked from the certification record. It must run against aarch64 repositories and fail if any package installed by the updater or a migration cannot be resolved. It must then boot a previous released image or package set, run the real non-interactive updater, and confirm the graphical login target. A hardware update sign-off still runs the candidate on the physical machine; CI does not replace it. This matrix is a platform-support claim, so merging this standard intentionally freezes all Omarchy-MLX release tags until the gate exists and is green.

## Release rule

Do not create any Omarchy-MLX release tag unless the Apple Silicon update gate is green, at least one SoC/board remains Certified, and every row marked **Certified** has a complete, passing hardware sign-off for the candidate SHA recorded in the certification file. If a Certified check fails, fix it and test a new candidate, or lower that row's tier for evidence-based reasons before tagging; record a tier change in both the certification file and release notes.

A failed or missing Supported check does not alone block a tag, but it must be recorded. For boot, display, GPU acceleration, update or rollback/recovery, treat an unverified result as a failure and lower the row to Experimental or Not yet before tagging. For another failure, the release maintainer records the keep-or-lower decision in the certification file. A sign-off on one SoC/board never substitutes for another; M4 becomes a required certification target only after its support work lands and its tier is promoted.
