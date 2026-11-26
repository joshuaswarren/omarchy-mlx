# SUP-320 - format bridge: SUP-300 sweep output -> SUP-266 gate-cap table

Main's SUP-320. SUP-266 staged the 13-inch G13G `sdpa_online_max_keys`
default-gate sweep kit (jwm1:~/scratch/wAY-sup266, kit.sha256 verified);
SUP-300 is the cp314 wheel that runs it on Spark. Spark dropped off the
network mid-build (~10:14 CST, host unreachable); the build re-runs clean
when the host returns (idempotent, 15G gate). So tonight there is NO
sweep output yet - this receipt is the FORMAT BRIDGE: the exact table the
sweep JSONL lands in, with the mechanical jq mapping from `sweep.py`'s
lines to each cell, and every value slot marked PENDING. When SUP-300's
wheel lands and the sweep runs, the numbers drop in and the gate decision
reads off the last row.

## Sweep output contract (sweep.py, verified on jwm1)

Per leg: `{"cell": [model, ctx], "arm": "A"|"B", ...decode/tps fields...}`
Per cell: `{"model": ..., "ctx": ..., "verdict": "IDS-IDENTICAL"|"IDS-DIVERGE",
"delta_pct": <(med(B)-med(A))/med(A)*100>, ...}` where arm A =
`MLX_OMARCHY_SDPA_ONLINE=0` (live default), arm B = `=1` (forced online).

## The formatted table (values PENDING until the sweep runs)

Grid per SUP-266 plan: models {2b SiddhJagani/Qwen3.8-2B-mlx-4Bit @0867d98b,
4b mlx-community/Qwen3-4B-2507} x ctx {512..4096 in the plan's 256 steps,
5-pair ABBA each}.

| model | ctx | A med tok/s | B med tok/s | delta_pct | verdict | cell meaning |
|---|---|---|---|---|---|---|
| 2b | 512 | PENDING | PENDING | PENDING | PENDING | |
| 2b | ... | PENDING | PENDING | PENDING | PENDING | |
| 2b | 4096 | PENDING | PENDING | PENDING | PENDING | |
| 4b | 512 | PENDING | PENDING | PENDING | PENDING | |
| 4b | ... | PENDING | PENDING | PENDING | PENDING | |
| 4b | 4096 | PENDING | PENDING | PENDING | PENDING | |

## The mechanical mapping (jq, runs on out/sweep-<utc>.jsonl)

Cell medians:
`jq -s 'map(select(.model)) | group_by([.model,.ctx]) | map({cell: .[0].cell, a: ([.[] | select(.arm=="A") | .tps] | sort | .[length/2|floor]), b: ([.[] | select(.arm=="B") | .tps] | sort | .[length/2|floor])})'`
(the tps field name is whatever sweep.py's leg line carries; verify against
the first leg line before the sweep - one-line check, no code change).

Cell verdicts + delta_pct: take the LAST cell line per (model, ctx).

## Gate decision rule (SUP-266 plan, restated for the table reader)

N = largest ctx where median delta is not negative (win or within noise).
First ctx with median delta <= -1% marks the regression edge. The
`sdpa_online_max_keys` cap = the largest ctx below that edge. Flat through
4096 = report flat; the gate follows the last measured cell, never an
extrapolation. IDS-DIVERGE on any cell invalidates that cell (re-run before
reading the cap).

## Provenance

- SUP-266 kit: jwm1:~/scratch/wAY-sup266 (plan.md sha256 4510437bd977...,
  sweep.py, sweep.sh, kit.sha256 - sha256sum -c verified on host).
- SUP-300 wheel: cp314 build on Spark, died mid-flight with the host drop,
  re-runs on return (w7H, journal 10:14 CST).
- G13C precedent for the cap decision: cap 2304 - per-layer KV at ctx4096
  16.8 MB x 4 head sharing = 67 MB > 48 MB SLC, forced online -7.25%.
