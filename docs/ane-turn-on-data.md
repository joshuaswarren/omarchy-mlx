# Turning the ANE on for an untested chip: what a deep row carries

The deep collector (`scripts/collect_deep.py`, payload schema v2) captures
everything the omarchy-ane lane needs to author an overlay and driver row
for an SoC nobody has run yet. This page lists what a row contains, what
each chip's community rows already cover, and the promotion rule for
turning a chip on by default. Nothing here names private hosts; every
field is queryable through
`python3 scripts/query_community_data.py show <sha>`.

## What a v2 deep row carries

Linux (`ane_port_detail` plus the new turn-on blocks):

- **Devicetree** — the full ane node(s) (MMIO reg, reg-names, IRQs,
  iommus with phandles resolved to DART paths, power-domains, status),
  every DART node, the ANE **mailbox** node (reg window, interrupts,
  status), the **reserved-memory** subtree (reg, no-map, iommu-addresses,
  compatible), the complete pmgr offset topology with `ane_cpu` /
  `ane_set*` pwrstate children, the AIC, structured boot provenance
  (`asahi,*` chosen properties, model, root compatible), and the booted
  DTB hash (or the explicit `needs root` reason; the collector never
  calls sudo). Node discovery is pattern-based (`ane`, `iop-ane`,
  `ascwrap`, `exclave`, `dart-ane`, `t8020` markers), so M3–M6
  generations are captured without code changes.
- **Runtime** — `/proc/iomem` ranges (ane/dart/pmgr), loaded-module
  version, filtered kernel log, and the **firmware placement** (names,
  sizes, sha256 under the known firmware roots; the bytes never leave
  the machine).
- **`runtime.omarchy_ane`** (own cap 48 KiB; when over cap the dmesg
  tail is dropped first, never check/module/smoke/dmesg_faults) — the
  promotion block: `machine_id` / `owner_id` (first 16 hex of sha256 of
  random per-install tokens at `~/.config/mlx-omarchy/`; never a
  serial/UUID/MAC), `check` (`omarchy-ane-check` exit, ready/FAILED,
  UNTESTED flag, output lines), `module` (`ane` / `ane_t6021` with
  version, srcversion, parameters), `firmware` (`/lib/firmware/apple/ane`
  hashes), `opt_in` (the `ane-*` keys of
  `/etc/omarchy-platform/dtb-overlays.opt-in`), `smoke` (always
  present: `{requested: false}` without `--ane-smoke`; with the flag
  it runs the packaged `omarchy-ane-smoke` runner — one
  JSON line on stdout, one `omarchy-ane-smoke: ...` line on stderr;
  exit 0 = all 20 calls bit-exact, exit 1 = ran with failures and the
  JSON is kept with `errors > 0`, exit 2 = unavailable for this SoC —
  and otherwise records `available: false` with the reason),
  `uptime_s`, the first 200 filtered kernel-log lines (160 B each) with
  the fault subset, `/proc/interrupts` samples (idle pair 10 s apart, a
  third after the smoke), package versions, and host facts.
- **Archive members** — a strip-list-cleaned dtc text dump of the booted
  tree (`ane-linux-dt.txt`), and an optional m1n1 ADT dump attachment
  (`--adt-dump FILE`, capped 2 MiB, supplied by a developer run).

macOS (`ane_port_detail.macos` plus the new turn-on block):

- The IOService plane capture (H11ANEIn instances with core count,
  hardware generation and firmware state, ane/dart-ane/mapper-ane nodes
  with reg ranges and interrupt specifiers as hex, pmgr block base, the
  SET-base candidate with `driver_window_confirms`).
- The **IODeviceTree-plane** whitelist dump (`dt_nodes`): name,
  compatible, reg, interrupts, interrupt-names, IOInterruptSpecifiers,
  segment-ranges, ane-type/subtype/id, die-id, die-ane-id, clock-gates,
  power-gates, iommu-parent, phandle, vm-base, vm-size, page-size, sids,
  bypass-15, instance, dapf-instance-0, dart-id, dart-options — binary
  values as hex. Whitelist, not blacklist. Pattern-based matching
  captures `iop-ane` / `ascwrap` (M3) and `ane,t8020` (M4) shapes.
- `ane_macos` (deep): the full-property IODeviceTree dump of every
  ANE-family node (identity keys stripped; the strip-list names are
  recorded), the AIC identity (max-irq, #interrupt-cells), pmgr ANE
  power rows and `*tunables*` properties, ANE driver classes (pattern
  walk over IOClass/CFBundleIdentifier), loaded ANE kexts (version,
  Mach-O size + sha256 when readable), OS-shipped ANE firmware images
  (path, size, sha256 of public paths only, never the files), the
  optional CoreML smoke (20-call tiny add model, min/median ms), and
  `sw_vers` / hardware identity. A pattern with no match records
  `{matched: 0}`; a capture that cannot be read records an explicit
  `unavailable` reason, never a silent omission.

Redaction: every string passes the shared Redactor; the dump strip list
removes `serial-number`, `unique-chip-id` (ECID), `mlb`, `mac-address`,
`*-uuid`, `boot-nonce`, `nonce-seeds`, `random-seed`, `fv-*` and
`*-hash` keys before anything is serialized. Raw dumps also live in the
8 MiB archive with sha256 in the payload; the payload itself is capped
(256 KiB wire limit) and records every drop.

## Coverage today (community rows, 2026-10-01)

Counts are published rows of each kind. "SET confirmed" = a macOS row
whose `set_base_candidate.driver_window_confirms` is true (the pmgr+0xc000
driver window). Mailbox / reserved-memory / firmware / `omarchy_ane` /
smoke are new in v2 — no published row carries them yet; every Linux
row so far also records `dtb_sha256: null` (`needs root`).

| SoC | Chip | Linux deep | macOS deep | quick | SET confirmed | DARTs | Mailbox | Notes |
|---|---|---|---|---|---|---|---|---|
| t8103 | M1 | 5 | 0 | 3 | n/a (SET known from m1n1) | yes | **missing** | driver qualified; smoke rows exist upstream |
| t6000 | M1 Pro | 4 | 0 | 2 | n/a | yes | **missing** | overlay emitted; needs passing smoke rows |
| t6001 | M1 Max | 3 | 0 | 3 | n/a | yes | **missing** | Linux ANE live; needs passing smoke rows |
| t6002 | M1 Ultra | 0 | 0 | 6 | **no macOS row** | — | — | quick only; dual-boot macOS deep capture is the single unblock |
| t6020 | M2 Pro | 13 | 5 | 22 | yes (macOS) | yes | **missing** | most-covered M2 SoC |
| t6021 | M2 Max | 2 | 1 | 3 | yes (macOS) | yes | **missing** | research driver opt-in |
| t6022 | M2 Ultra | 0 | 1 | 0 | yes (macOS) | — | — | needs a Linux deep row for high bits + board topology |
| t8112 | base M2 | 2 | 1 | 0 | unmeasured | partial | **missing** | ASC tunables / chip-revision question (own-memory mode) |
| t8122 | M3 family | 0 | 0 | 1 | no | — | — | M3 = `iop,ascwrap-v6` ASC IOP; capture lands with v2 |
| t603x / T8132+ | M4–M6 | 0 | 0 | 0 | no | — | — | `ane,t8020` (M4), exclave (M5), dual `ascwrap-v8` (M6); v2 patterns cover them |

Every chip's gap list shrinks to the same v2 fields: **mailbox,
reserved-memory, firmware hashes, the `omarchy_ane` promotion block, and
a smoke result**. One Linux deep run + one macOS deep run on any
untested machine now supplies all of them.

## Promotion rule (mirrors the omarchy-ane README, which is authoritative)

> A SoC moves from opt-in to on-by-default after **one passing community
> smoke row** and no unresolved failing row. One passing row plus one failing
> row on an opt-in SoC is **CONFLICT**; do not promote until the failure is
> explained or superseded.
>
> A passing row must have:
> - `check.exit == 0` and `check.status == "ready"`;
> - `installed == true` and `module.name` equal to the chip's driver module;
> - an attempted 20-call add-fixture smoke with every output bit-exact and
>   `errors == 0`;
> - no ANE/DART/mailbox fault line in `dmesg`.
>
> The collector runs the smoke only when load1 is `< 0.5` and CPU PSI
> `avg10 == 0.00`. It polls every 5 seconds for up to 300 seconds. If the
> machine remains busy, it records every load/PSI decision and sets
> `smoke.attempted == false`; the row is not judged, not failed.
>
> Regression: for an on-by-default chip, a latest unclean row can revert it
> to opt-in. Rows without an installed driver or an attempted smoke are not
> judged. A failing smoke row remains a failure; an unavailable or busy smoke
> is not a failure.

The checker needs `received_at`, `check.{exit,status}`, `installed`,
`dmesg_faults`, `dmesg`, `module.name`, `machine_id`, `owner_id`, the
smoke result and its attempted state, boot compatible, and kernel. The
collector also records `uptime_s` as host context, but it is not a gate.
`smoke.idle_checks` records `load1`, `psi_cpu_avg10`, and the idle decision
for every poll. The post-smoke `/proc/interrupts` sample is recorded but
not judged.

`driver_source` says how the bound ANE module was produced: `intree` when
the module file lives in the kernel package's tree
(`.../kernel/drivers/accel/ane/`) or the module is built in (listed in
`modules.builtin`, or `/sys/module/<m>` present with no file anywhere);
`dkms` when it lives under `updates/`, `extra/`, or a dkms path; `none`
when no ANE driver is bound. `driver_source_evidence` keeps the proof:
the module file path (home paths and hostnames redacted) and the `builtin`
flag. `dtbs_source` reads the `DTBS=` line of
`/etc/default/update-m1n1` read-only: a non-empty value is `kernel` (m1n1
boots the kernel's own DTBs, so the overlay opt-in has no effect and only
the kernel DT node can enable the chip), empty or absent is `overlay`, a
missing file is `unknown`. Per issue #155, a row with
`driver_source == "intree"`, the chip's driver bound, a 20/20 bit-exact
smoke, and no faults is judged exactly like a dkms row; the omarchy-ane
checker (w73) owns that verdict and its wording. The untested-chip steps
the collector prints branch on both facts: on an in-tree kernel it says
userspace + smoke + firmware fetch only and never omarchy-ane-dkms, and
when `dtbs_source == "kernel"` it appends that the overlay opt-in has no
effect.

Smoke fixture availability (omarchy-ane 50958be): the H14-family chips
t6020 / t6021 / t6022 / t8112 run `fixtures/h14-anec/add` (20,800 B,
sha256 `b416b9d1...`, mil-hwx-compiler output, encoder
h14-oracle-parity). M1-family (t8103 / t6000 / t6001 / t6002) currently
exit 2 — no H13 (M1-family) add fixture has run through
`omarchy-ane-run` yet. Next steps on the omarchy-ane side: commit the
mil-hwx-compiler rung-1 add package as `fixtures/h13-anec/add`, run it
20x via `ane-run --check add` on a T8103 or T6001, and add the H13
chips + golden to the smoke.

Send corrections or an override only through the omarchy-ane lane
(w73); this file describes the collector side.

### Linux probe detail

Deep Linux rows also carry ane_linux.ane_probe, the read-only JSON from
omarchy-ane-probe --json (15-second timeout, 8 KiB cap). It adds SoC,
board/model/compatible identity, ANE/DART/pmgr/mailbox topology, kernel
and command-line allowlist, modules and interrupts, accelerator/platform
devices, power domains, debug state, package/install state, filtered dmesg,
check and firmware results, SoC-table comparison, unreadable reasons and
elapsed time. See omarchy-ane's
[docs/ane-probe.md](https://github.com/joshuaswarren/omarchy-ane/blob/main/docs/ane-probe.md)
for the complete field contract. It complements the existing omarchy_ane
promotion block; it does not replace it.
