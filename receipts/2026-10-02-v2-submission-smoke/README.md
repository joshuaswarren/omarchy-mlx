# v2 community-data submission smoke: blocked by a redaction finding — no row submitted

Lane: SubmitV2. Pre-registered private notebook entry + raw artifacts with SHA256SUMS
back every claim here (private lab; host placeholders below).

## Status

**BLOCKED before submit.** The v0.7.12 deep collector on the M1 Max Omarchy host
(T6001) produced a fully valid schema v2 payload, but the redaction grep found the
host's short fleet alias surviving inside journal-quoted systemd unit paths. Per the
task's defect rule, the submission was NOT made: zero rows were sent, the live service
was never written to, and no workaround was applied.

## What ran

All commands on the M1 Max Omarchy host (placeholders; `<url>` is the documented
service origin):

```bash
# scripts/ from tag v0.7.12 (== commit 8995ddf2e7d463a9ffc30854da21f74f632768bd),
# deployed via git archive; collector sha256 e90d0c4a3f2fe6995f9a4db5e9750cd4721c136106610ea1374da642ade1f808
python3 scripts/collect_deep.py --out /var/tmp/SubmitV2/preview.tar.gz   # preview, no upload
# in-process dry build of the exact submit payload (run_sections → assemble_files →
# finalize with the upload step omitted) → payload-inspect.json
# SUBMISSION STEP INTENTIONALLY NOT RUN — see Defect
```

- Preview: exit 0, ~11 s. `schema_version: 2`, `sections_unavailable:
  [correctness, benchmark, profile]` (no mlx wheel installed — expected), archive
  23828 bytes, sha256 `37b34b4a3b16c3e94f4e7a2fa2108530c0e595043e7b80d0dad744ebaaf8fc53`.
- Post-state verified: `ane` module load state and module parameters identical before
  and after; no reboot, no GPU use; host footprint limited to `/var/tmp/SubmitV2`.

## Schema v2 field presence (assembled payload, `kind: deep`)

| Field | Present | Value / shape |
|---|---|---|
| `schema_version` | yes | `2` |
| `ane_linux` | yes | available; full deep block |
| `ane_port_detail` | yes | full structure incl. devicetree |
| `…runtime.omarchy_ane.machine_id` | yes | derived 12-hex token |
| `…runtime.omarchy_ane.owner_id` | yes | derived 12-hex token |
| `…runtime.omarchy_ane.check` | yes | available, exit 0, status `ready` |
| `…runtime.omarchy_ane.module` | yes | name/params/srcversion/version |
| `…runtime.omarchy_ane.firmware` | yes | explicit unavailable: no firmware dir |
| `…runtime.omarchy_ane.uptime_s` | yes | integer |
| `…runtime.omarchy_ane.dmesg_faults` | yes | 1 fault line |
| `…runtime.omarchy_ane.smoke` | yes | `{requested: false}` (see note) |
| `devicetree.mailbox` | yes | present |
| `devicetree.reserved_memory` | yes | 25 children |
| `devicetree.dtb_sha256` | null | with `dtb_sha256_error: "needs root"` — collector never sudo, as required |

Smoke note: without `--ane-smoke` the released contract is `{"requested": false}`
(`test_collect.py:2240`). The `{available: false, reason: "…not shipped in omarchy-ane
yet"}` shape is produced only with the opt-in flag. Not a code defect; the task's field
list quoted the opt-in shape.

## Redaction check (counts only)

Greps over the assembled payload and every archive member:

| Pattern | Hits |
|---|---|
| full hostname | 0 |
| user name | 0 |
| LAN prefixes (both home subnets) | 0 |
| MAC address pattern | 0 |
| Tailscale CGNAT address / literal | 0 |
| `serial-number` property | 0 |
| other identity substrings (user first/last, host model aliases, peer hostnames) | 0 |
| **host short fleet alias (strict prefix of the hostname)** | **3** |

The 3 hits are the same journal line quoted in three places (`ane_linux.dmesg`,
`ane_port_detail.runtime.dmesg`, `…omarchy_ane.dmesg`): systemd validating a
host-admin-created unit file whose name embeds the host alias. The Redactor replaced
the hostname token with `[host]` (47 replacements) but its rule is a whole-token match
on the full hostname (`scripts/collect_common.py:143-147`), so a derived alias inside a
path never matches. The service-side PII scan mirrors the collector patterns
(`services/community-data/src/pii.ts`), so the row would have been accepted and
published with the alias.

## Defect

Host-alias survival in redaction: any identifier derived from the hostname (admin-chosen
unit names, path components) passes both the collector redactor and the service PII
scan. Suggested fix: redact hostname-prefix aliases (e.g. also replace the hostname's
leading token before a `-`/`.` boundary) and mirror it in `pii.ts`; add a regression
fixture quoting a unit path. Optional hardening: ship the dry-build payload inspection
used here (`finalize` minus upload) as a `--dry-build` flag so any contributor can grep
before consenting.

## Not done (blocked by the defect)

No submission, hence no row sha256, no live `GET /v1/results/<sha256>` round-trip, no
`query_community_data.py` visibility check, no `ane_soc_from_collect.py` run on a
submitted row. The private notebook entry carries the exact next discriminator; rerun
this procedure unchanged once the redactor fix lands.
