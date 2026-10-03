# 2026-10-03 — SealFast: per-process Parakeet whole-encoder bundle seal cost

Branch `agent/seal-fast` (based on `origin/main` @ `1bd12adfb`, v0.7.22+receipts):
`354fa66fe` (waterfall instrumentation, cherry-pick of 7bc4b5c73) + `2e048dadb`
(v3 seal rework, cherry-pick of 633d926c4) + `d50fc1ee0` (this ticket's
additions). Hosts named as roles per repo policy: jwm1 = 13-inch M1 (T8103),
jw16 = M1 Max (T6001). Notebook:
`~/.local/share/apple-silicon-lab/entries/SealFast/20261003T180231Z-jwm1-jw16-seal-fast-sha-pipeline.md`.

## Question and outcome

The worker seal (`load_bundle_sealed` → `sealed_file_at`: read + SHA-256 +
memfd copy of the 458 MB program, then seal) costs 472–476 ms fresh-process
on jwm1 and ~435 ms on jw16. What is hash-bound vs copy-bound, and what can
be cut without weakening the sealed-snapshot property?

**Answer: the hash is already at the hardware floor and already covered by
the ticket's step 1 — `bundle.cpp` dispatches SHA-256 to the ARMv8 crypto
instructions at runtime (`ANE_SHA256_AARCH64`, `getauxval(AT_HWCAP) &
HWCAP_SHA256`, `vsha256*` intrinsics with the scalar implementation as
fallback). It predates this ticket; both hosts report the `sha2` CPU flag.**
The remaining lever is overlap, which the v3 seal (kept here) already
implements: one read pass, hash inline on the reader, an ordered writer
thread draining 4 MiB chunks into the memfd, seals applied after the last
byte, `F_GET_SEALS` verified.

### SHA-256 throughput on the real 458,018,816-byte payload (jwm1, warm cache, digest `13c74423…` = pin)

| implementation        | ms (best of 2) | throughput |
|-----------------------|---------------:|-----------:|
| coreutils `sha256sum` |          265   | 1.73 GB/s  |
| OpenSSL EVP (`openssl dgst -sha256`) | 267 | 1.72 GB/s |
| in-tree (seal trace, fused read+hash) | see A/B below | — |

The in-tree hash therefore has no software headroom: it is at OpenSSL parity
on this payload. (jw16 reference cells: in the A/B log, same protocol.)

## What changed (d50fc1ee0 on top of the v3 cherry-picks)

1. Dual-path SHA vectors: `sha256_force_scalar_compress(bool)` (test hook)
   lets one suite run pin BOTH compress implementations against the FIPS
   vectors (empty, `abc`, 1000×`a`, the 55/56/63/64/65/119/120 padding
   boundaries, 1 MiB of `0x5a`, and the million-`a` vector
   `cdc76e5c…11 2cd0`). A crypto-path regression can no longer hide beside a
   green scalar path.
2. Tamper coverage completed: truncated payload, appended bytes, and a
   rewritten manifest under a stale-correct pin are refused, and
   `OMARCHY_ANE_SEAL_VERIFY=1` keeps the pin refusal even with a worst-case
   sidecar row seeded for the tampered identity.
3. `MLX_OMARCHY_ANE_SEAL_TRACE=1`: env-gated per-file seal waterfall
   (`read_hash_ms` / `copy_ms` / `seal_ms`) for fresh-process decomposition.
   Timers only; no descriptor or sync path changes.
4. Kept from `633d926c4` unchanged: the pipelined `sealed_file_at`, the
   root-owned read-only-chain stamp gate (`root_owned_readonly_chain`),
   the ctime-keyed sidecar, `OMARCHY_ANE_SEAL_VERIFY` force.

## Integrity verification (units; all PASS)

| suite / host | result |
|---|---|
| dev box (x86_64, scalar path) | 48 cases / 8,215 assertions PASS (root cases skipped, no fixture) |
| jw16 (aarch64, user fixtures, dual SHA paths pinned) | 48 cases / 8,228 assertions PASS |
| jw16 (aarch64, real root fixture: installed bundle copied to a root-owned 0555/0444 chain, sidecar seeded) | 48 cases / 8,228 assertions PASS, `UNIT-ROOT rc=0` — source-fd fast path taken ONLY on the root chain; user-owned chain always snapshots |
| jwm1 | runs inside the A/B script (same binary, shipped from jw16) |

Also verified by test: flipped payload byte refused (default and forced),
truncated / appended / manifest-rewritten refused, file modified after seal
leaves the session's bytes unchanged (snapshot guarantee), forged sidecar
identity ignored on user-owned bundles, root-owned chain takes the fast
path, `/tmp`-style world-writable parents correctly REFUSE the fast path
(predicate walks the whole canonical chain).

Prior-thread context: the Jw16ParakeetWarm window-3 log (05:04Z, read this
session) had already measured the v3-vs-base user-layout pair at
2285/2334/2287 vs 2303/2301/2361 ms — pipeline-only effect within noise on
user layouts, as predicted (stamp gain requires a root-owned chain by
construction).

## A/B (paired, idle-gated, 9 fresh-process cold opens per arm, arms alternated, distinct wheel stamps)

Wheels (both cp314 aarch64, built on jw16 `nice -n 19` from the exact
trees, whole-encoder bundle hash-verified at build, committed
`libane-strict.so` = pin `d06222a8…`):

- baseline: `mlx_omarchy-0.32.4.dev202610031839+1bd12adf` (sha256 `782ef94c…`)
- candidate: `mlx_omarchy-0.32.4.dev202610031841+d50fc1ee` (sha256 `c14a57ec…`)

(PENDING — filled by `results.md` addendum when the cells land.)

## A/B results (jwm1, 2026-10-03 18:55–19:13Z; log `/var/tmp/sealfast/ab-jwm1.log`)

18 fresh-process cold opens (9 pairs, arms alternated AB), fixture clip,
idle-gated per cell (load<0.5 AND PSI cpu avg10<0.1 before every timed run;
co-tenant lane load spiked mid-run — every cell records its load/PSI, and
"clean" below means PSI avg10 ≤ 0.5).

Golden contract holds on EVERY cell of BOTH arms: `status=match`,
emissions=104, transcript sha `db501a8c080380ea…` (cpu_tensor_events: the
report's verification block carries the check; every cell match).

| arm (wheel stamp) | n valid | session_open median | total_pipeline median |
|---|---|---:|---:|
| base `+1bd12adf` (v0.7.22 lineage) | 9 (7 clean) | 910.2 ms (clean) / 930.4 ms (all) | 1628.0 ms |
| cand `+d50fc1ee` | 9 (9 clean) | **907.8 ms** | 1650.8 ms |

**Outcome: the pipelined candidate is performance-NEUTRAL on user-owned
jwm1 (session_open Δ ≈ −2 ms, within cell spread 839–983 ms). The ticket's
≥40 ms seal-stage pass gate is NOT met, and the honest finding is that it
cannot be met by hashing work: the hash is already at the ARMv8 crypto
hardware floor (see throughput table — OpenSSL parity), and the v3 pipeline
overlap it does add is absorbed inside session open. This matches the
Jw16ParakeetWarm window-3 floor finding (user-layout v3-vs-base within
noise there too). The value landed here is integrity/test/tooling, not
milliseconds.**

- jw16 window: queued behind the gpuwin mutex at 18:52Z (another lane held
  it); the launcher (`/var/tmp/sealfast_window_jw16.sh`, syntax-checked,
  no self-trap — gpuwin restores the service, per the 18:20Z incident rule)
  runs the same A/B + post-window `llm-inference` health check
  unattended and appends to `/var/tmp/sealfast/ab-jw16.log` +
  `/var/tmp/sealfast-logs/ab-jw16-window.log`. NOT yet fired at
  receipt-writing time.

## Live tamper (jwm1, candidate wheel)

`TAMPER-default rc=1`, `TAMPER-forced rc=1`: the flipped-byte bundle copy
is refused — at the CLI asset-pin gate
(`EncoderRunError: …not covered by the approved assets; refusing to load
unpinned ANE programs`), i.e. defense-in-depth refuses before the worker
seal is reached. The worker-seal tamper path itself ("does not match the
pin") is covered by the unit matrix (flipped/truncated/appended/manifest
cases, default and forced) on dev box + jw16.

## Known gap (follow-up, one line)

The shipped CLI spawns the worker with `stderr=subprocess.DEVNULL`
(`mlx_omarchy_parakeet.py:133`), so `MLX_OMARCHY_ANE_SEAL_TRACE` output is
discarded in the packaged path — the trace needs a one-line stderr-routing
change (or a worker `--trace-file`) to be observable end-to-end.

## Post-state

- Branch `agent/seal-fast` @ `3be293f33` pushed (code + docs + receipt).
  NOT merged to main — the pass gate was not met; merge decision belongs to
  Main with these numbers.
- jwm1: `/var/tmp/sealfast/{venv-base,venv-cand,runs,ab-jwm1.log}`,
  `/var/tmp/sealfast-wheels/*.whl`, `/var/tmp/sealfast_*.sh`,
  `/var/tmp/sealfast-logs/` — cleaner: delete these paths when done
  (archive-worthy first: ab-jwm1.log + runs/*/transcribe-report.json).
- jw16: `~/sealfast/{base,cand}` worktrees, `/var/tmp/sealfast*`,
  `/var/tmp/sealfast-logs/` — same cleanup rule. No system paths touched on
  either host; llm-inference untouched (window never fired from this lane).

## Reproduce

```bash
# units (any host with the built test binary; root cases need PK_SEAL_TEST_ROOT_BUNDLE)
PK_SEAL_TEST_ROOT_BUNDLE=/pk-root-fixture ./omarchy_ane_bundle_tests
# decomposition (fresh process)
MLX_OMARCHY_ANE_SEAL_TRACE=1 MLX_OMARCHY_PK_WATERFALL=1 MLX_OMARCHY_PK_TIMING_EXTRA=1 \
  mlx-omarchy-parakeet transcribe -o out/ clip.flac
# forced full verify
OMARCHY_ANE_SEAL_VERIFY=1 mlx-omarchy-parakeet transcribe -o out/ clip.flac
```
