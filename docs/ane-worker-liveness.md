# ANE worker liveness: which receipts' worker counts mean nothing

`tools/ane_worker_liveness.py` is the only correct reader in the tree. Snapshot
code that answers "is a worker still running?" with pgrep is wrong in one of
two ways, both measured on m1-test-host-class Linux with a live stand-in worker
(`tail` copied to the worker's name):

| form | live worker present | no worker, a shell merely names it |
| --- | --- | --- |
| `pgrep -c -x mlx-omarchy-ane-worker` | **0** (never matches) | 0 |
| `pgrep -cf mlx-omarchy-ane-worker` | 1 | **1** (its own shell) |
| `pgrep -a mlx-omarchy-ane` | 1 | 0 (correct) |
| `pgrep -c -x mlx-omarchy-ane` | 1 | 0 (correct) |
| `tools/ane_worker_liveness.py` | 1 | 0 (correct) |

`-x` compares the whole `/proc/<pid>/comm`, and TASK_COMM_LEN is 16 bytes, so
comm holds 15 characters plus a NUL: `mlx-omarchy-ane`. The 22-character full
name can therefore never equal it. `-x` does not reject the long pattern; the
string it is compared against is short. Spelled against the truncation
(`-x mlx-omarchy-ane`) it is correct in both directions. `-f` matches any
command line containing the string, including the caller's own shell and the
driver that launched the worker.

Every comm-matching form shares one residual weakness: any unrelated process
whose own comm truncates to those 15 bytes reads as a worker. That is the safe
direction for a "no workers" claim, since it over-reports rather than missing a
live worker, but it does not separate a real worker from a stand-in. Only the
`/proc/<pid>/fd` half does that: a real worker holds `/dev/accel/accel0` open
and a stand-in does not, which is why `device_holders` is reported next to
`worker_count` instead of in place of it.

The two failures are not equally dangerous. A `-c -x` zero reads as "nothing
found" and invites a second look, so it corrupts a receipt by omission. A `-cf`
one from the naming shell looks exactly like a healthy worker and passes
review, so it corrupts a receipt by assertion. Where the self-match sits
decides its direction: in a pre-submit clearance gate it fails closed, which
refuses to start on a clean box and is survivable; as a post-run health probe
the same expression fails open and reports a worker that does not exist.
Any `-f` check needs to exclude its own pid and its parents' before its answer
means anything; matching `basename(argv[0])` sidesteps that instead of
patching it.

`ps -ww` matters as much as the match rule: without `-ww`, ps truncates the
command column at 80 characters into a pipe, which cuts long argv[0] paths and
loses the worker.

## Receipts whose `workers` field is meaningless

Always-0 readings from `pgrep -c -x` in
`receipts/2026-09-14-encoder-islands-exec-m1-test-host-linux/derivation/run_islands.sh:23`
(branch `receipts/2026-09-14-encoder-split-plan`):

- `receipts/2026-09-14-encoder-islands-exec-m1-test-host-linux.json` — the four
  snapshot `workers: "0"` values, and `workers_after: 0`, which is those
  snapshots hardcoded forward in `derivation/build_receipt.py:210`.
- `receipts/2026-09-14-encoder-islands-exec-m1-test-host-linux/device-out/state-pre.txt`,
  `state-post.txt`, `state-post-B.txt`, `state-post-C.txt` — `workers=0` (plus
  a stray `0` line, because `pgrep -c` exits 1 on a count of zero and the
  `|| echo 0` fires too).

Both directions, from the same snapshot function reused with a `-cf` post pass:

- `receipts/2026-09-14-encoder-parity-ane/device-out/state-pre.txt`
  (`workers=0`) and `state-post.txt` (`workers=3` with no worker alive), and
  the copies embedded in `receipts/2026-09-14-encoder-parity-ane/environment.json`
  and `receipts/2026-09-14-encoder-parity-ane.json`. That receipt already
  refuses both readings and carries the ps plus `/proc/*/fd` value instead;
  the raw lines stay meaningless.

Sound, and not on this list: every receipt using `pgrep -a mlx-omarchy-ane`
(the 2026-09-13 t6001-test-host/m1-test-host soak, utilization and full runs, the t6001-test-host exec
receipts, and `receipts/2026-09-14-encoder-islands-exec-t6001-test-hostmbp1-linux.json`).
The prefix form matches the truncated comm, so it does detect a live worker.

Over-counting but not meaningless: `pgrep -af '^.*/mlx-omarchy-ane-worker([[:space:]]|$)'`
(the ANE runtime-bridge qualification scripts) matches a live worker, and also
any process whose command line merely contains the worker path, such as the
driver that spawned it.
