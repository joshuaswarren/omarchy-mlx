# ANE matmul work rates on three Apple Silicon laptops, Linux driver (2026-10-18)

This receipt backs the work-rate table in `docs/architecture.md`. Every timing
below is a Linux driver measurement from 2026-10-18: sealed constant-weight
matmul programs at M=64, one `ane-run` process per rung, 200 calls, three
rounds, the submitting process pinned to one performance core, the host ANE
queue exclusive (GPU queue paused) during each run, `flock` on the run lock.
Busy per job is `ane_stats` busy_ns delta over the jobs delta of the same
rung. Output correctness is checked on every rung: byte-exact `sha256` on the
two M1 chips, and the 2 ulp or 4 condition unit band checker on the M2 chip
(H14 target). All runs are in the `raw/` files of this receipt.

## Identity

| Machine | Kernel | ANE driver | Runner | Programs |
|---|---|---|---|---|
| 13-inch M1 (T8103) | `7.1.12-2-20261013.18-sep-ARCH`, boot `<id>` | stock `ane` DRM driver, in-tree | `ane-run` sha256 `480040b5f700f407f11563f5349e7614df9fefe0efe57327c68ebc35b9779805` | H13 set E seal `94aff857fadcd20d9c5bb8ad1e87cf31706fc5a2603a544cffefb254dc4237d1` |
| 16-inch M1 Max (T6001) | `7.1.12-2-12.6-sep-ARCH`, boot `<id>` | stock `ane` DRM driver, in-tree | same as above | same H13 set E seal |
| 14-inch M2 Max (T6021) | `7.1.12-2-20261013.18-sep-ARCH`, boot `<id>` | `ane_t6021` (sha of the loaded module's source tree verified at run time by the queue pins) | `ane-run` sha256 `943ae8c67598175a542f9a2c370852d60ba9a810cb1e8db11fed92efc2daf2e1` | H14 grid seal `2b69285e0c293cd346908491b5f3e0e6b982570e2159769426701e92f7eccf57`; H14 scale seal `ad6ea1cf55f453db7ca95ab5a7efea5298d2e66addeef61dfc6e52e0ebe365ca` |

Program provenance: the seals hold the compiler output, the deterministic
inputs and weights, the expected outputs, and per-rung manifests. The H14
seals' Apple records are cross-target compiler output used for byte parity of
the task streams and constants; they are not runtime measurements, and no
timing in this receipt comes from them. All timing comes from the Linux runs
above. The MIL, deterministic inputs and dense expected outputs of the 268 M
and 537 M rungs are the same across the H13 and H14 targets by construction
of the two seals; program bytes differ between targets by design.

## Commands

Inside each rung directory (names in the raw files):

```
flock /var/tmp/ane-run.lock taskset -c <pinned core> timeout 120 \
  ane-run --anec program.anec --in 0=in0.bin --out 0=out0.bin \
  --repeat 200 --time          # H13 rungs; exact sha check on out0.bin after
flock /var/tmp/ane-run.lock taskset -c 5 timeout 120 \
  ane-run --anec program.anec --in 0=x.f16 --out 0=y.f16 \
  --weights w.raw.f16 --repeat 200 --time   # H14 rungs; check_rung.py after
```

Pinned core: cpu7 on the 13-inch, cpu2 on the 16-inch, cpu5 on the M2 Max.
Timing procedure: three rounds per rung, back to back, in one boot per
machine; no thermal clamp; the queue gate refuses to start unless no other
ANE or GPU process runs. Each machine was verified on its stock boot with a
zeroed ANE counter before its first run of the day, and no reboot was
required after any run in this receipt (all runs are read-only submits).

## Results, busy per job in ms (raw per-round values)

| Work | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| 268 M MACs (K2048 N2048) | 0.2151, 0.2161, 0.2152 | 0.5966, 0.6524, 0.6130 | 0.8069, 0.8087, 0.8088 |
| 537 M MACs (K4096 N2048) | 0.4003, 0.3998, 0.4009 | 1.0710, 1.1256, 1.0146 | 1.1166, 1.1108, 1.1088 |
| 4.2 M MACs (K256 N256), M2 only | not run | not run | 0.2872, 0.3261, 0.3259 |
| 67 M MACs (K1024 N1024), M2 only | not run | not run | 0.3752, 0.3977, 0.3873 |

Exec medians agree with busy within 0.4 percent at the two larger sizes on
the M2 Max; the H13 runner's exec median at 537 M was 0.405 to 0.408 ms
(13-inch) and 0.993 to 1.139 ms (16-inch). Output checks: 18 of 18 exact on
the 13-inch (six rungs over the day including the 8.4 M and 16.8 M exchange
pairs), 6 of 6 exact on the 16-inch, 33 of 33 in band on the M2 Max (27 grid
plus 6 scale). Wrong-lane, NaN, short-output and nonzero-padding cases are
rejected by the checkers; a one-bit mutation of an H13 expected output is
rejected by the sha check.

Recovery: all runs are measure-only submits. Each host's ANE counter,
`lsmod` state and boot id were read after the last run of the day; no module
was loaded on the two M1 laptops at any point, and the M2 Max kept its stock
`ane_t6021` with `fw_perf_mode` N and a clean allocator.
