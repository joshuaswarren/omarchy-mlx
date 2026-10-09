# Forks and the backport flow

mlx-omarchy patches four upstream lineages. Each has a fork under
`joshuaswarren` carrying our work on branches. Work lands in the fork first.
Upstreaming a change is a separate, later decision.

## Upstream to fork map

The lineage rows below preserve their original fork bases; the current
build consumes `mlx.lock` and the patch set described below.

| Upstream | Fork | Branch | Base and carried work |
| --- | --- | --- | --- |
| `ml-explore/mlx` | `joshuaswarren/omarchy-mlx` | `omarchy` | MLX 0.32.2 commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, the `mlx.lock` pin, plus the six patches from `patches/`, one commit per patch |
| `eiln/ane` and `allbilly/libane` | `joshuaswarren/omarchy-ane` | `omarchy`, `omarchy-kmd` | `omarchy`: eiln/ane tip `0dcea99` plus the six-commit libane series from `~/keep/eiln-ane-series/`, applied with `git am`. `omarchy-kmd`: allbilly/libane head `1e0afd8` plus the m1-test-host debug instrumentation from `~/keep/ane-kmd-local/local-kmd-changes.patch` |
| `AsahiLinux/linux` | `joshuaswarren/linux` | `ane-dt-t8103` | Asahi tag `asahi-7.1.6-1`, the exact source of the Arch kernel `7.1.6-1-1-ARCH` on the M1 test machine, plus two commits `326d6033059d18a1f47833ba7ad3a3ee2c4eb443` and `b1cb024a1`: the ANE device-tree node, `status = "disabled"` in `t8103.dtsi` and enabled in `t8103-j293.dts`. After hostile review found that an enabled ANE DART node would let the in-tree `apple-dart` driver and the out-of-tree driver fight over the same MMIO window and IRQ 417, the second commit keeps the ANE DART node disabled everywhere and drops the `iommus` phandle. |
| `gitlab.freedesktop.org/mesa/mesa` | `joshuaswarren/mesa-1` (current work); `joshuaswarren/mesa` (historical reproductions) | `honeykrisp-omarchy`; historical `honeykrisp-miscompile-repros` | New driver work targets `mesa-1`. The reproduction branch remains in the older repository as of 2026-09-20; it must not be linked as though migrated. |

`eiln/ane` and `allbilly/libane` are one lineage. `allbilly/libane` is a fork
of `eiln/ane` that stays ahead of it, and it carries both the kernel module in
`ane/` and the userspace library in `libane/`. GitHub allows one repository
per account per fork network, so one fork serves both. The clone carries two
upstream remotes: `upstream` for `eiln/ane` and `allbilly` for
`allbilly/libane`. Do not try to fork `allbilly/libane` separately; GitHub
refuses a second repository in the same network.

The instrumentation on `omarchy-kmd` is captured from an uncommitted working
tree on m1-test-host. It is not a finished fix, and the module currently resets the
machine when loaded.

`allbilly/ane` is documentation and register research with examples. We depend
on it as an upstream reference. We do not patch it, so it has no fork.

The former `joshuaswarren/omarchy-libane` and
`DeckardAndFriends/omarchy-libane` repositories are archived. Each README
points at `joshuaswarren/omarchy-ane` as the live repository for this lineage.

## Why we fork

Owner decision, 2026-09-03: maintain forks, backport from upstream into them,
and add our fixes there. We do not wait on upstream review to land work.
Whether to upstream a change stays open per change. Two standing rules:
never push to an upstream remote, and never force-push.

## Backport flow

Example for `omarchy-ane`. The same steps work in every fork clone:

1. `git fetch upstream` for `eiln/ane`, or `git fetch allbilly` for `allbilly/libane`
2. `git checkout <branch>`
3. `git merge <remote>/main` for a full sync, or `git cherry-pick <commit>` for a targeted backport
4. `git push origin <branch>`

Rules:

- Upstream history is never rewritten. Merge or cherry-pick. Never rebase a shared branch onto upstream, because that would demand a force-push.
- Never force-push.
- All pushes go to `origin`. The upstream remotes are read-only in practice.

`omarchy-mlx` starts at the `mlx.lock` pin, not at the upstream tip. A merge of
`upstream/main` also brings everything after the pin. To ride one upstream fix
on the pinned base, cherry-pick that commit instead.

## Deliberately not wired up

The build still consumes the pinned upstream archive:

- `mlx.lock` pins MLX 0.32.3 at commit `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`, with the archive URL and SHA-256 (verified against main on 2026-09-30).
- `scripts/prepare-mlx.sh` downloads the archive and applies the patches from `patches/` (58 files after the 2026-10-09 upstream backports #4658 #4656 #4657 #4649).
- Nothing in the build reads `joshuaswarren/omarchy-mlx`.

Switching the build to the fork is an owner decision that has not been made.
It would change the release story: a release would name a fork commit instead
of an upstream archive plus patch files, the patch set would live as commits
instead of `patches/*.patch`, and provenance would point at `joshuaswarren`
instead of `ml-explore`.

## ane.dtbo

`allbilly/libane` ships `ane.dtbo`. A full static device tree that carries an
ANE node cost this project seven of eight CPU cores. An overlay applied to the
live tree is the correct mechanism. Read `docs/boot-and-kernel.md` before
kernel work that touches the device tree.

## Kernel device tree fork

`joshuaswarren/linux` exists so the ANE device-tree work has a reviewable
home. Mesa will not take user-space driver support until a kernel driver is
upstream, so kernel-side work is prepared here first. The branch
`ane-dt-t8103` carries the device-tree patch that lets the ANE node reach
the kernel through the normal m1n1 payload path instead of a GRUB
`devicetree` override, which freezes m1n1's per-boot patching and cost the
test machine seven of its eight CPU cores. See `docs/boot-and-kernel.md`
for the failure and the verification checklist.

The branch has not been booted. It has no DT binding document and no
in-tree driver; both are explicitly named at the repository-root cover
`ANE-STAGING-NOTE.md` on the branch (commit `9457669`). The shipped
`ane.dtbo` in `joshuaswarren/omarchy-ane` is also documented there as
defective (`ane@23b100000` is the AIC address on `t8103`; its `iommus`
and `power-domains` are unresolved `0xffffffff` fixups), independent of
whether the node shape in `ane-dt-t8103` is right. The out-of-tree driver
in `joshuaswarren/omarchy-ane` must stay blacklisted; loading it
hard-reset the M1 on 2026-09-03. Do not pair this device-tree change
with an automatic module load.

## Mesa miscompile reproductions

New driver branches and PRs target `joshuaswarren/mesa-1` for upstreaming
toward `omacom/mesa`. This does not relocate historical branch URLs.
On 2026-09-20, `git ls-remote` confirmed `honeykrisp-miscompile-repros`
at `7302d43288b1aa2b561dcd1b9dba0b88109e12ae` in
[`joshuaswarren/mesa`](https://github.com/joshuaswarren/mesa/tree/honeykrisp-miscompile-repros),
not in `mesa-1`. It carries standalone Vulkan reproductions of the five
Honeykrisp miscompiles. Historical upstream is `gitlab.freedesktop.org/mesa/mesa`;
the fork was cut from `intel-lgci-fdo-gitlab-mirror/mesa.mesa` on 2026-09-03.

Purpose: Dj does the Mesa-side Apple GPU work and asked for the fixes
in a form he can read, evaluate, and reuse on newer ISAs. The branch
carries one reproduction per miscompile, expected-versus-observed
numbers with the hardware and driver version, the shipped mlx-omarchy
workaround per defect with receipt links, and the family analysis.
The family claim - dynamic byte extraction, a data-dependent shift
feeding a mask, covering cases 2, 3, and 5 - was confirmed on Apple
hardware on 2026-09-03: the case-3 matrix arms with the dynamic shift
fail on m1-test-host while the constant-shift controls pass on the same
device, and the selector is refuted as the trigger. Cases 1 and 4 do
not reproduce on that build and stay on the branch labeled as misses
with both readings. The README in `src/asahi/repro/` on the branch is
the authority; this file only maps the fork.

Limits, stated on the branch itself: the shaders are reconstructions
of the receipts' probe shapes, not copies; the misses are not yet
re-derived; the llvmpipe verification proves the programs compute the
expected values on a conformant driver, and the hardware run supplies
the rest.
