# 2026-10-06 — mlx-omarchy v0.7.29 release preparation, phase A

Status: phase A complete; phase B waits for Main's GO. Process inherited
unchanged from `receipts/2026-10-05-release-0.7.28/README.md` and
`receipts/2026-10-05-golden-build/README.md`: same gates, same standards
(provenance, compiled statements, asset sha verification). Candidate cut base:
main `78470402f` (CPU PD hold); pending landings may move it. Companion
evidence (private lab notebook): entry H332 and
`artifacts/Release0729/2026-10-06-jw16-udev-ship/`.

Hosts are named by chip class only, per repo policy (no hostnames/IPs in
committed files): "T6001 laptop" = the G13C M1 Max verification laptop,
"T8103 host" = the G13G M1, "T6021 gate host" = the M2 Max release-gate box.

## Phase A results

### 1. The udev rule ships through the existing packaging path (no PKGBUILD package() change needed)

`install_system()` in install.sh (block "PM-QoS access for the CPU deep-idle
hold") installs the rule to
`<dest-root>/usr/lib/udev/rules.d/70-omarchy-mlx-cpu-dma-latency.rules`; the
omarchy-pkgs PKGBUILD `build()` runs
`install.sh --system --dest-root "$srcdir/stage"`, and
`package_omarchy-mlx()` copies the whole staged tree
(`cp -a "$srcdir/stage/usr" "$pkgdir/usr"`). The rule therefore ships with the
omarchy-mlx package automatically. The v0.7.28 tag predates the rule — it
ships from v0.7.29 because the tag tarball's install.sh is what the PKGBUILD
runs.

Fresh-install simulation on the T6001 laptop (worktree 78470402f, v0.7.28
wheel + vendor assets, the newest published; re-run at the tagged sha is a
phase-B gate): staged tree contains
`usr/lib/udev/rules.d/70-omarchy-mlx-cpu-dma-latency.rules`, sha256
`d63915c47b8247c7a95f5babd4b3318a6590b73c66d74a05c6c311d3e9e4622b`,
byte-identical to `packaging/udev/70-omarchy-mlx-cpu-dma-latency.rules`
(provenance: `sha256sum` on repo file, staged file, and installed file — all
three equal; full transcripts in the lab notebook artifacts).

### 2. Activation at install time is missing — prepared `.install` for the phase-B PR

`pkgbuilds/omarchy-mlx/` carries no `.install` (dir listing: `PKGBUILD`,
`.omarchy` only), so without one the rule applies at the next udev event or
reboot. Prepared text (phase B adds `omarchy-mlx.install` next to the
PKGBUILD):

```bash
# Apply the CPU PD hold udev rule immediately, no reboot (the hold is
# packaged with omarchy-mlx; see overlay/mlx/backend/omarchy/cpu_pd_hold.h).
# On package removal the device node keeps the applied permissions until the
# next reboot — standard udev behavior, harmless (0660 root:video).
post_install() {
  command -v udevadm >/dev/null 2>&1 || return 0
  udevadm control --reload || true
  udevadm trigger --subsystem-match=misc --sysname-match=cpu_dma_latency || true
  udevadm settle || true
}
post_upgrade() { post_install "$@"; }
```

Verified live on the T6001 laptop: with the rule installed,
`udevadm control --reload && udevadm trigger --subsystem-match=misc
--sysname-match=cpu_dma_latency` (plus `udevadm settle`) applied it with no
reboot: `/dev/cpu_dma_latency` went `root root 600` -> `root video 660`
(`stat -c '%G %a'`), and an open+write as user with supplementary group
`video` flipped from `Permission denied` (baseline negative control, captured
before any change) to `OK`.

Behavior note from the same run: removing the rule and re-triggering does NOT
revert the node (udev never rolls back applied modes; a trigger with no
matching rule is a no-op). Harmless for removals — the node keeps
`0660 root:video` until reboot — but the manual restore is part of the
recorded procedure below.

### 3. Chip gate of the hold (G13C excluded by default; rule is inert there)

`cpu_pd_hold_enabled_for` (overlay/mlx/backend/omarchy/cpu_pd_hold.cpp; the
header documents: unset env -> on for device names containing `G13` but not
`G13C`; `0` -> off; any other value -> forced on) was compiled standalone from
the REAL translation unit on the T6001 laptop
(`g++ -std=c++17 -O0 -ffunction-sections -fdata-sections` with
`-Wl,--gc-sections`, mlx headers from the laptop's venv — unreferenced
functions GC'd, so only the pure decision function links; command and source
in the lab notebook artifacts) and printed:

```
Apple M1 Max (G13C C0)     default=0 forced1=1 off=0   <- T6001 laptop (mlx-omarchy-info device name)
Apple M1 (G13G)            default=1 forced1=1 off=0   <- T8103 host class
Apple M2 Max (G14C B1)     default=0 forced1=1 off=0   <- T6021 gate host class
```

On the G13C laptop the hold is OFF by default even with the rule installed —
the rule is inert there by design (matches the 78470402f commit title, "G13
(non-C) parts"). The live fd mechanism (fd open while GPU work is in flight,
closed < 1 s after idle, `=0` never opens) was proven on the T8103 host on
this same commit (lab notebook H331: `tests/test_cpu_pd_hold.py` 2/2, digests
identical in every A/B pair). The fd probe could not run on the T6001 laptop
because its installed wheel (`0.32.4.dev202610061642+7bb9f86`, presence of
the hold checked via `git cat-file -e 7bb9f86:overlay/.../cpu_pd_hold.h` —
absent) predates the hold, and no >= 78470402f wheel exists outside the
T8103 lane; the phase-B release battery repeats it on the release wheel
(gate g-hold).

### 4. Exact changes made on the T6001 laptop — and restored

- Added `/usr/lib/udev/rules.d/70-omarchy-mlx-cpu-dma-latency.rules` (sha256
  `d63915c4…`, copied from the staged tree) — REMOVED after the probes.
- `udevadm control --reload` + `udevadm trigger --subsystem-match=misc
  --sysname-match=cpu_dma_latency` + `udevadm settle`, twice (apply, and
  after removal).
- Node restore after rule removal (udev does not revert): `chown root:root`
  + `chmod 600` on `/dev/cpu_dma_latency`; re-verified `root root 600` and
  the video-member open denied again — byte-for-byte the pre-run baseline.
- Throwaway staging left on the laptop (recursive deletion is blocked by this
  session's shell policy; /tmp is tmpfs and clears at reboot; safe to delete):
  `/tmp/rel0729-src` 1.3 GB, `/tmp/rel0729-stage.4jJZ` 838 MB,
  `/tmp/holdgate` 48 KB, `/tmp/holdgate-main.cpp` 16 KB.
- No packages installed/removed, no services touched, no $HOME changes, no
  GPU jobs (the laptop's running omlx server was untouched; the GPU lock was
  never needed).

## v0.7.29 phase-B checklist (execute on Main's GO)

Preconditions: MoeLayer2 Sub kernel, allocator fence fix, SDPA chunk fix
landed on main; re-read the tracked issue and the current upstream-watch
digest before the cut.

1. Cut: `git fetch origin`; annotated tag `v0.7.29` at the agreed main sha;
   push the tag BEFORE any gate runs; the tag never moves (0.7.28 discipline).
2. Golden/release build on the macstudio OrbStack VM, exactly per
   `receipts/2026-10-05-golden-build/README.md`: glslang 2026.3 / shaderc
   1.4.357 pins (the VM's pacman glslang/shaderc must match; use the rescued
   `/opt/m2-tc` 2026.3 prefix or the dev-box glslc 2026.3 tarball); SPIR-V
   spot-checks `bc6eb65b…` (matmul_f32_coopmat_qk) and `fb361c32…`
   (gather_qmm_sub_bf16).
3. Wheel: `mlx_omarchy-0.32.4.dev<UTC compact>+<7-sha>-cp314-cp314-linux_aarch64.whl`;
   aarch64-only again (the 0.7.28 x86 decision stands unless Main reopens it).
4. Vendor tar: `packaging/vendor-wheels.sh` against
   `packaging/requirements-lock.in` -> `omarchy-mlx-vendor-wheels-v0.7.29-cp314-aarch64.tar`
   (+ `.sha256`); verified by `packaging/verify-vendor.sh`.
5. Release gates g1–g15 per the 0.7.28 table on the T6021 gate host (fresh
   throwaway GATE_ROOT), plus the v0.7.29-specific gates:
   - **g-udev**: at the tagged sha, re-run `install.sh --system` staging and
     require the staged rule sha256 == `d63915c4…`.
   - **g-hold** (T8103/G13G host — book through the orchestrator; that host
     belongs to the power-experiments lane): `tests/test_cpu_pd_hold.py` 2/2
     on the release wheel with the rule installed; decode digests bit-identical
     to the standing pins in the same window (the hold may change timing,
     never output: 2B `eee1cf96…`, 4B `42d27a8cbe93df49…`, 9B `80274aa7…`).
   - **g-g13c** (T6001 laptop): default gate still 0 and no cpu_dma_latency fd
     during a GPU job on the release wheel; `mlx-omarchy-info` name unchanged.
   - **g-omlx-convfuse**: `test_omlx_q35_conv_fuse.py` 4/4 GREEN on the
     patched venv (patch 0006, commit `bf62cfb65`, is on main and ships in
     the tag tarball under `packaging/omlx-linux/patches/`); kill switch
     `MLX_OMARCHY_OMLX_CONV_FUSE=0` documented.
6. Draft release with wheel + vendor tar + `.sha256` + final SHA256SUMS
   (3-asset coverage; replace any provisional file before publish).
7. `scripts/verify-release-assets.py v0.7.29 --platforms linux_aarch64` on the
   draft -> `VERIFIED`, rc=0; publish `--draft=false --latest`; re-verify the
   published release; publish-time install test from the published URL in a
   fresh HOME (installer smoke + one short generation).
8. omarchy-pkgs bump PR (permitted; NO PRs/issues/comments to ml-explore,
   jundot/omlx, ashhart/*, AsahiLinux): `pkgver=0.7.29`, `pkgrel=1`,
   `_wheel=<exact built wheel name>` (keep the `${_wheel//+/%2B}` URL
   escaping), sha256sums in order: tag tarball, wheel, vendor tar, mesa
   `e7631595df…` (unchanged unless the vulkan side moves — then the driver
   sha + tarball sha update together), ADD `omarchy-mlx.install` from
   section 2. External omarchy-pkgs PRs need the `build-approved` label
   (Ryan Hughes / Spencer Bull / Bjarne Øverli) — file the PR early; the
   label gates the `[omarchy]` repo.
9. Land the release receipt and this file on main by fetch+rebase (never
   force); update user-facing docs if the hold changes install text.

## Blockers for phase B

1. Main's GO (the three pending landings land first).
2. A T8103-lane slot for g-hold (request through the orchestrator; never
   touch that host directly from this lane).
3. Nothing else: the conv-fuse patch is already on main, the PKGBUILD needs
   no package() change, and aarch64-only repeats by default.

## Provenance

| claim | value | how obtained |
|---|---|---|
| rule file sha256 | `d63915c47b8247c7a95f5babd4b3318a6590b73c66d74a05c6c311d3e9e4622b` | `sha256sum` repo + staged + installed (3/3 equal) |
| staged tree contains rule | yes | `find <stage> -name '*.rules'` after `install.sh --system` on the T6001 laptop |
| live permission flip | `root root 600` -> `root video 660` -> restored `root root 600` | `stat -c '%G %a'`; open probe via `setpriv --groups video` (denied / OK / denied) |
| gate decision | G13C 0 / G13G 1 / G14C 0, forced 1, off 0 | compiled from cpu_pd_hold.cpp (`--gc-sections`), executed on the T6001 laptop |
| T6001 laptop device name | `Apple M1 Max (G13C C0)` | `mlx-omarchy-info` output |
| installed wheel on T6001 laptop | `0.32.4.dev202610061642+7bb9f86`, no hold | dist-info listing; `git cat-file -e 7bb9f86:overlay/mlx/backend/omarchy/cpu_pd_hold.h` (absent) |
| fd mechanism on G13G | live test 2/2, digests identical | lab notebook H331 (same commit); NOT re-run here |
| SPIR-V spot-check shas `bc6eb65b…`, `fb361c32…` | inherited pins | golden-build receipt; NOT computed in phase A |
| decode digest pins `eee1cf96…` / `42d27a8c…` / `80274aa7…` | recorded pins | 0.7.28 receipt; NOT computed in phase A |
