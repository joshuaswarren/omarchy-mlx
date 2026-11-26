# v2 community-data submission smoke: one accepted v2 row, live-verified

Lane: SubmitV2. Pre-registered private notebook entry + raw artifacts with SHA256SUMS
back every claim here (private lab; host placeholders below).

## Status

**COMPLETE.** One schema v2 deep row from the M1 Max Omarchy host (T6001) is accepted
and served by the live service, verified field-by-field through the public API.
Blocked once by a host-alias redaction leak (see first receipt in this directory's
history / the fix commit), fixed, then resubmitted clean.

## Provenance

- Leak fix: commit `0afee96eb` on main ("collect: redact hostname-derived aliases in
  paths and mirror the scan server-side") — collector substring redaction for
  hostname-derived aliases + request-only alias header the service scans against.
  Suites: collector python 169 OK (128 deep + 41 macOS; also fixed a latent
  `__main__`-block placement that silently skipped 24 tail tests in the documented
  direct run), service bun 81 OK (78 + 3 new), privacy hook clean (3 re-pinned
  synthetic fixture blobs).
- Submitted scripts: git archive of the fix commit; collector `collect_common.py`
  sha256 `7236b318a68ec18ea55d005e1de24834f5b72f9b9ef1072fbebb49ffd6ddfe62` verified
  identical on the host before the run.
- Pre-flight: preview + in-process payload dry build on the host — alias grep 0,
  `redaction_summary` shows the new `hostname_alias: 1` counter working.
- Benchmark-timing rule: this row carries NO timing numbers (no mlx wheel installed;
  correctness/benchmark/profile sections record explicit `available: false`), so no
  load/PSI gate applies to any number here.

## Submission

- Command (placeholders): `python3 scripts/collect_deep.py --out …/submit.tar.gz
  --submit <service-origin>` — passing `--submit` is the documented consent.
- Row sha256 / content address:
  `66091f89cef62b35e6215c68404ef93a872568810c36ffbf560f7d002219db75`
- Receipt URL: `<service-origin>/v1/results/66091f89cef62b35e6215c68404ef93a872568810c36ffbf560f7d002219db75`
- Upload answer: stored, `deduplicated=false`.

## Live round-trip (GET /v1/results/<sha>, custom User-Agent)

| Field | Sent | Served |
|---|---|---|
| `schema_version` / `kind` | 2 / deep | 2 / deep |
| `chip` / `model` / arch, kernel | apple,t6001 / MacBook Pro (16-inch, M1 Max, 2021) / aarch64 | identical |
| `ane_linux.available` | true | true |
| `runtime.omarchy_ane.machine_id` / `owner_id` | derived 12-hex tokens | identical |
| `runtime.omarchy_ane.check` | exit 0, status ready | identical |
| `runtime.omarchy_ane.module` | ane 0.2.0.r14.g87f427f + params | identical |
| `runtime.omarchy_ane.firmware` | explicit unavailable (no firmware dir) | identical |
| `runtime.omarchy_ane.uptime_s` / `dmesg_faults` | integer / 1 line | identical |
| `runtime.omarchy_ane.smoke` | `{requested: false}` (no `--ane-smoke`; documented contract) | identical |
| `devicetree.mailbox` / `reserved_memory` | present / 25 children | identical |
| `devicetree.dtb_sha256` | null + `dtb_sha256_error: "needs root"` (never sudo) | identical |
| `ane_linux.interrupts` | 2 idle phases × 14 lines | identical |
| alias grep over served row | — | 0 |

`ane_soc_from_collect.py <row.json>`: runs, groups the row under t6001, and REFUSES
overlay generation with the documented gate — the SET candidate only comes from a
macOS deep row, and this is a Linux-only row. The Linux side it did ingest is sane:
ane node `soc/ane@…` (`apple,t6000-ane`), `interrupts = <0 0 770 4>` — decimal AIC
encoding (IRQ 770, flags 4), no hex garbage; `/proc/interrupts` samples carry decimal
IRQ numbers. ane0/die-0 overlay pick is pending a macOS row by design, not a defect.

## Consumer notes (observed, not defects)

- `query_community_data.py --source remote --json --kind deep --chip t6001 list`
  shows the row (4 t6001 deep rows). The task's literal `--chip M1` matches nothing:
  the filter matches SoC/model strings (`apple,t6001`), not marketing names. Full
  unfiltered list: 43 rows.
- The dataset mirror (`/v1/dataset/latest.jsonl`, 97 rows) intentionally strips
  `ane_port_detail` (store.ts keeps per-row detail only at `/v1/results/<sha>`);
  `ane_linux` itself round-trips through the mirror.

## Redaction counts (submitted row)

hostname 47, hostname_alias 1 (the new counter), ipv4 10; user names, MACs, UUIDs,
serials, LAN/Tailscale addresses: 0 occurrences in the served row. Smoke field spec
corrected in `docs/ane-turn-on-data.md`: without `--ane-smoke` the field is always
present as `{requested: false}`.
