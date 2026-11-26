# Perf hold: power-profiles-daemon is not viable on Apple Silicon — feasibility evidence and ranked alternatives

- Date: 2026-10-02
- Actor: PerfHold (worker); ticket: hold the CPU governor at `performance`
  for the duration of each inference request via the ppd D-Bus
  `HoldProfile` API, no root (mechanism requested after the w71 lane's
  H202 arm data: performance governor lifts TTFT rate +6.6-8.4% and
  prefill +1.5-1.8%, decode unchanged, on the M1 host).
- Verdict: **ppd cannot deliver this mechanism on these hosts, at any
  authorization level.** Implementation stopped per ticket; no code
  shipped; no A/B run (it was contingent on feasibility).
- Notebook (private, source of record):
  entries/PerfHold/20261002T085434Z-jw16-ppd-perf-hold-feasibility-ab.md,
  artifacts/PerfHold/20261002T085434Z-jw16-ppd-feasibility/ (SHA256SUMS
  inside).

## Feasibility table

| Check | M1 Max host (T6001, "jw16") | M1 host (T8103, "jwm1") |
|---|---|---|
| power-profiles-daemon | 0.30-1, active, enabled | 0.30-1, active, enabled |
| Profiles offered | `balanced`, `power-saver` — both `PlatformDriver: placeholder`; **no `performance`** (daemon `Profiles` property: `aa{sv} 2`) | identical: placeholder-only pair, **no `performance`** |
| ACPI platform_profile | absent (`/sys/firmware/acpi/platform_profile: No such file`) | absent |
| cpufreq | `apple-cpufreq`; policies 0 (E) / 2, 6 (P); `performance` IS in `scaling_available_governors`; all schedutil | `apple-cpufreq`; policies 0 (E, max 2064 MHz) / 4 (P, max 3204 MHz); all schedutil |
| Hold `performance`, user (ssh session) | `AccessDenied: Not Authorized: org.freedesktop.UPower.PowerProfiles.hold-profile (9)` (polkit: `allow_active` only) | not probed at user level; same policy file ships on the same package |
| Hold `performance`, root (polkit-authorized) | `InvalidArgs: Cannot hold profile 'performance' as it is not available (16)` | same package/daemon shape; to be re-run post M2-window for the row |
| Hold `power-saver`, root (authorized, profile exists) | hold ACCEPTED (rc 0) and **no governor moved** — policy0/2/6 schedutil before/during/after (placeholder driver is a cpufreq no-op) | to be re-run post M2-window |
| `powerprofilesctl set performance` | invalid choice (client list = power-saver/balanced only) | same |

Three independent, fatal layers on Apple Silicon + ppd 0.30:

1. **No `performance` profile exists.** ppd offers only profiles some
   driver claims; the packaged driver set is `placeholder` only (no
   ACPI platform_profile on Asahi, no apple-cpufreq driver in ppd 0.30).
   Even a fully authorized `HoldProfile('performance')` fails
   daemon-side (`InvalidArgs`, quoted above).
2. **Holds are polkit-gated to seat-active sessions**
   (`allow_active=yes`, `allow_inactive=no`, `allow_any=no` in
   `power-profiles-daemon.policy`). A serve running under ssh or a
   systemd user service is denied; only processes inside the active
   graphical session could hold. (Demonstrated: same hold, same
   profile, user ssh session → AccessDenied; root → accepted.)
3. **Even successful holds move no cpufreq state.** Root-authorized
   `launch -p power-saver` held for 6 s: zero governor/frequency
   changes on any policy — the placeholder driver does nothing, so a
   hold cannot change the governor even in principle on these hosts.

Also verified: holds auto-release when the launching process exits
(after every probe the profile returned to `balanced`; nothing stuck).
So the lifecycle design would have worked — the mechanism cannot.

Idle rule: every probe ran at loadavg 0.00-0.04, PSI cpu avg10 0.00
(values in artifacts). Provenance line n/a: no MLX engine loaded; these
are daemon/sysfs probes. The alternative-lever proof below flipped
policy6 `schedutil → performance → schedutil` in 1.3 s and restored it.

## Ranked alternatives (no root at request time, after a one-time install)

**A (recommended): narrow sudoers rule + tiny root-owned governor helper.**
A root-owned script (e.g. `/usr/local/lib/mlx-omarchy/perf-governor`)
with exactly two modes — `performance` (write `performance` to every
`/sys/devices/system/cpu/cpufreq/policy*/scaling_governor`, remember the
prior values) and `restore` — plus a sudoers drop-in permitting exactly
that path with NOPASSWD. The serve-side helper keeps the planned shape
(shared module, refcount so overlapping requests hold once, silent
debug-only fallback when unavailable, env off-switch); the subprocess
backend is `sudo -n <helper> performance|restore` at refcount 0↔1.
Proven end-to-end on T6001 in this session: sysfs flip through sudo
works (07-flip-revert-proof.txt). Cost: one helper + one sudoers file
per host (one-time root install, ~30 lines total); w71 already drives
these governors with root manually. Risk: low and auditable (command is
a fixed allowlisted path); leak window if the serve is SIGKILLed while
holding leaves the governor at `performance` — benign on these single-user
laptops (that is H202's PERF state; self-heals at boot; no
overheat mechanism demonstrated at these clocks) — and recoverable by
the next successful release call. Scope the sudoers rule tighter than
this host's current broad `NOPASSWD` grants.

**B: udev rule making `scaling_governor` group-writable.** One boot-time
udev rule (`RUN+=` chgrp/chmod on `policy*/scaling_governor`) plus a
dedicated group; serve writes sysfs directly, no sudo at runtime. Cost:
similar to A, but needs a reboot (or udevadm trigger) to activate,
grants every process of the group permanent write access (coarser than
A's two verbs), and loses A's per-change logging. Viable fallback.

**C (root cause for the original ppd design): ship a ppd driver for
apple-cpufreq.** A small ppd platform driver that claims `performance`
on apple-cpufreq hosts would make `HoldProfile('performance')` real and
set governors — the requested mechanism, unmodified, with polkit doing
its job for desktop-launched serves. Cost: a C plugin packaged against
ppd's plugin ABI and kept current across ppd updates (ppd just dropped
its old cpufreq driver in 0.30); days, not hours; upstreamable to the
omarchy package. Note it still only helps seat-active clients (layer 2
is ppd policy, not our bug). Right home: the omarchy packaging lane, not
this repo.

**D: root systemd socket-activated helper service.** A tiny unit
exposing a local socket that flips governors on request. No broad sudo,
cleanest privilege story, but a new long-lived component to maintain on
each host — over-built for two personal laptops given A exists.

**E: sudoers rule for `cpupower`.** Same shape as A but through
`cpupower frequency-set`, whose CLI also accepts `-d/-u` bounds and
other verbs — a custom two-verb helper (A) is strictly narrower to
allowlist. A only.

Not viable without one of the above: ppd as-is (evidence above);
`omarchy-powerprofiles-set performance` (it checks
`profile_available performance` and falls back to balanced — same
missing profile); direct sysfs writes from the serve (root:root 0644).

## What was NOT done

- A/B TTFT/prefill measurement on both hosts: contingent on
  feasibility; it failed, so no numbers were captured (and none are
  claimed). The w71 H202 entry remains the evidence that the governor
  lever itself is real (+1.8% prefill, +8.4% TTFT rate, digests exact).
- No serve code, tests, docs, or env var: stopped per ticket instruction.
- jwm1 authorized-probe rows (root holds, flip proof): scheduled after
  the M2-window-end broadcast because jwm1 currently hosts the M2 proxy
  lane; its read-only rows are captured and identical to T6001's.

---

# Addendum (2026-10-02, same session): the alternative lever, measured on both Macs

Supersedes the "NOT done" rows above: Main redirected the ticket — the
recommended alternative was to be quantified by flipping the governor
exactly as the helper would (root sysfs write of `scaling_governor`,
revert verified), A/B against the stock governor on each Mac. The ppd
verdict above stands unchanged.

## Method (identical on both hosts)

- Engine: published release wheel v0.7.14
  (`mlx_omarchy-0.32.4.dev202610020636+07729f40-cp314-cp314-linux_aarch64.whl`,
  sha256-verified against the release `SHA256SUMS`) in a private
  `/tmp` venv; `scripts/mlx_provenance.py`: `verified: match` on both
  hosts before any run.
- Harness: w71's qwen38 protocol (`qwen38-mlx-bench.py` +
  `qwen38_bench_lib.py`, corpus `qwen38-2b-prompts.jsonl`, model
  Qwen3.8-2B-mlx-4Bit). One cell = 10 prompts, 512-token pure-prefill
  leg + 32 greedy decode tokens, warmup 2, passes 1.
- 5 pairs, order off,on,on,off,off,on,on,off,off,on; idle gate before
  every cell (load < 0.5 AND PSI cpu some avg10 = 0, recorded); the
  flip is `echo <gov> | sudo -n tee /sys/devices/system/cpu/
  cpufreq/policy*/scaling_governor` — exactly the write the proposed
  helper performs — with a governor readback check per cell and a
  verified final revert. Nothing persistent was installed on either
  host (venv, model copy, script all under /tmp; the helper itself was
  NOT installed — the test uses the same sysfs write through sudo).
- Idle/thermal/provenance recorded per cell; GPU access on the M1 Max
  inside one `gpuwin` window (llm-inference stopped, flock
  /tmp/m1-gpu.lock, service restored: health_ok=1,
  probe_finish=length); the M1 host under `flock /tmp/m1-gpu.lock`.

## Numbers (medians; percent change ON vs OFF; greedy digests identical in every cell of every run)

| Host (chip) | run | ttft_tok_rate | pure_prefill_tok_rate | decode_tok_rate | digests | thermal max |
|---|---|---|---|---|---|---|
| jw16 (T6001, M1 Max) | v1 governor, warm alternation | OFF 52.94 → ON 53.89 (**+1.79%**) | 138.17 → 140.67 (**+1.81%**) | 62.99 → 62.96 (−0.05%) | identical | 24.4 °C |
| jwm1 (T8103, M1) | v2 governor, decay-gated | OFF 41.57 → ON 43.09 (**+3.66%**) | 61.57 → 62.20 (**+1.02%**) | 29.19 → 29.18 (−0.03%) | identical | 32.6 °C |
| jwm1 | combo: uclamp_min=1024 (unprivileged) | 42.03 → 43.16 (**+2.69%**) | 61.51 → 62.14 (**+1.02%**) | 29.19 → 29.12 (−0.24%) | identical | 32.3 °C |
| jwm1 | combo: taskset P-cores | 41.59 → 42.36 (+1.85%) | 61.51 → 61.92 (+0.67%) | 29.20 → 29.17 (−0.10%) | identical | 33.4 °C |
| jwm1 | combo: governor | 42.19 → 42.91 (+1.71%) | 61.51 → 62.21 (+1.14%) | 29.17 → 29.17 (0.00%) | identical | 33.2 °C |
| jw16 | combo: uclamp_min=1024 | 53.08 → 53.22 (+0.26%) | 137.61 → 140.71 (**+2.25%**) | 62.97 → 62.97 (0.00%) | identical | 25.2 °C |
| jw16 | combo: taskset P-cores | 52.65 → 53.42 (+1.46%) | 137.54 → 140.56 (**+2.20%**) | 63.00 → 62.97 (−0.05%) | identical | 26.2 °C |
| jw16 | combo: governor (4 pairs) | 53.12 → 52.62 (−0.93%) | 140.84 → 138.51 (−1.65%) | 62.94 → 62.93 (−0.02%) | identical | 26.3 °C |

- **Digests**: `ordered_records_sha256` IDENTICAL in every cell of every
  run — and identical across BOTH hosts, BOTH arms, and every route:
  `1aa2f5f8aa1f5741027b42d067ba6897dae2465820067de9c16d8a2529179e7c`
  (same wheel, model, corpus, greedy temp 0). No lever affects outputs.
- **Decode** unchanged everywhere (±0.24% worst), matching H202.
- **The ≥ ~6% TTFT gate is NOT met on either host in today's machine
  state — for any route, including the performance governor itself**
  (jw16 governor: +1.79% / −0.93% across two runs — within noise).
  Measured mechanism for the shortfall: the P cluster NEVER decays to
  its floor on these hosts right now — `scaling_cur_freq` before every
  cell read 3036000 kHz (jw16) / 1956000-3204000 kHz (jwm1) under the
  stock governor, in every single cell, with the decay gate firing 0/28
  times. w71's +8.4% ttft-rate (H202) and −8.6% ttft latency (H204-209,
  uclamp) were measured against an idle-decayed P cluster that pays a
  frequency/placement ramp at request start; that state did not occur
  on either host during these runs (co-tenant lane load and/or
  apple-cpufreq's idle floor pre-pay the ramp). What survives today:
  prefill +1.0-2.3% for the placement routes, ttft +0.3-2.7%, decode
  and outputs bit-identical, zero regressions.
- Placement probe (main thread, /proc/<pid>/task/<pid>/stat processor
  field): with uclamp_min=1024, ALL samples land on P cores (jw16:
  cpu2-9, 38 samples; jwm1: cpu4-7, 40 samples). The off-arm sampler
  caught too few samples to be conclusive (bench exits mid-loop) and
  observed no E-core placement either — noted as a harness weakness,
  not evidence of absence.

## What shipped in this ticket (route 1, smallest privilege footprint)

`MLX_OMARCHY_UCLAMP_MIN` (default 1024; `0` disables; any other integer
sets the clamp) — implemented as an unprivileged in-process
`sched_setattr(SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_UTIL_CLAMP_MIN)` on
the serve's startup thread before the HTTP server and decode threads
spawn (threads inherit the clamp), with silent fallback on `EPERM`/
`EINVAL`/unsupported architecture (one stderr line, serving unchanged)
and no capability required (probed: works as the user on both hosts'
kernels, `CONFIG_UCLAMP_TASK=y`).

- `serve/mlx_omarchy_serve/perf_placement.py` — the clamp helper
  (stdlib ctypes; syscall table aarch64 274 / x86_64 314; refuses to
  guess unknown architectures).
- `serve/mlx_omarchy_serve/_mlxlm_server.py` — startup hook wired
  before `server_module.main()`, covering the OpenAI-compatible serve
  and every managed worker (they are the same server).
- `tests/test_perf_placement.py` — 9 tests (env switch off/default/
  explicit/garbage, flags+value in the packed attr, EPERM silent
  failure, unsupported arch, apply-from-env wiring). All pass; the
  live call was smoke-tested on jwm1 (applies, re-applies at 512).
- `docs/install-omarchy.md` — "Serving placement" paragraph.

Found and fixed during bring-up: `sched_setattr` takes THREE arguments
(pid, attr, syscall-flags) — the first draft omitted the trailing
`flags=0` and got EINVAL everywhere; the fixed call was verified live
on jwm1/aarch64. The dev box (PVE x86_64 kernel without uclamp groups)
returns EINVAL and exercises the silent fallback.

Privilege footprint of the shipped route: **none** — no sudoers, no
helper, no unit, nothing installed; the serve clamps its own threads.
The sudoers+helper design below remains the documented fallback for a
kernel that refuses the syscall, and the uclamp lever is strictly
safer than the governor lever it replaces: it is per-thread placement
policy, not a machine-global performance lock (other processes keep
the stock governor and can still idle).

## Fallback route (only if a future kernel refuses sched_setattr): narrow sudoers + two-verb helper

One-time root install per host; nothing at request time beyond two
fixed verbs:

1. `/usr/local/lib/mlx-omarchy/perf-governor` — root-owned 0755 shell
   script with EXACTLY two verbs:
   - `performance`: save current `scaling_governor` values, then write
     `performance` to every `/sys/devices/system/cpu/cpufreq/policy*/
     scaling_governor` (the write proven above);
   - `restore`: write the saved values back.
   Any other argument is refused. It can write no other file.
2. `/etc/sudoers.d/mlx-omarchy-perf-governor`:
   `joshuawarren ALL=(root) NOPASSWD: /usr/local/lib/mlx-omarchy/perf-governor performance, /usr/local/lib/mlx-omarchy/perf-governor restore`

Runtime contract in the serve: a refcounted hold (overlapping requests
hold once) calls `sudo -n <helper> performance` on 0→1 and
`sudo -n <helper> restore` on 1→0, inside try/finally, with a silent
debug-only fallback when the helper or sudo is absent — requests then
simply run at schedutil and are never blocked or failed by the hold.

Failure behavior:
- helper/sudoers missing or sudo denied → debug log, zero request
  impact;
- serve SIGKILLed mid-hold → governor stays `performance` until the
  next restore call or reboot (boots reset to schedutil; benign on
  these single-user laptops — that is H202's PERF arm state; max
  observed under full load here: 32.6 °C);
- the hold is machine-global for its duration (single-user hosts; GPU
  access already serializes through flock /tmp/m1-gpu.lock);
- scoped far tighter than the hosts' existing broad `NOPASSWD` grants:
  no other command, no options, no arbitrary paths, no boot
  persistence.

## Artifacts (private notebook)

entries/PerfHold/20261002T085434Z-jw16-ppd-perf-hold-feasibility-ab.md
and .../20261002T092000Z-jwm1-governor-ab.md;
artifacts/PerfHold/20261002T0910Z-jw16-governor-ab-v1/,
.../20261002T0920Z-jwm1-governor-ab-v1/,
.../20261002T0932Z-jwm1-governor-ab-v2/,
.../20261002T1024Z-jwm1-combo/,
.../20261002T1114Z-jw16-gov-noop/,
.../20261002T1124Z-jw16-combo/ (per-cell JSON + summary + SHA256SUMS
each).

## Honest limits

- The ≥ ~6% ttft gate was not met on either host during this session's
  runs; the shipped default (on) rests on: direction-consistent
  prefill gains (+1.0-2.3%), zero regressions anywhere, bit-exact
  outputs, zero privilege, and w71's controlled H204-209 result
  (−8.6% ttft latency) for the condition (idle-decayed P cluster)
  that today's co-tenant load never produced. If Main prefers the
  clamp OFF until the idle-state win is demonstrated in a quiet
  window, `MLX_OMARCHY_UCLAMP_MIN=0` is the one-line default change.
- The off-arm placement sampler was too coarse to independently
  document E-core placement on these hosts; the mechanism evidence
  rests on w71's H204-209 plus my on-arm P-only placement samples.
- Only the two Omarchy Apple Silicon hosts were probed; ppd's
  uselessness here is Apple-Silicon-specific (no platform_profile,
  placeholder driver) and does not generalize to x86 laptops.

## Addendum (2026-10-02, reported by the jwm1 lane; notebook entry H219; not re-measured here)

w71's independent jwm1 run on the v0.7.15-lane release venv measured the
uclamp lever against an idle-decayed P cluster — the machine state this
ticket's own A/B never hit — and saw the full effect this ticket could
not produce: pf512 first-token 0.1424 to 0.1303 s (**−8.7 %**), ttft rate
**+10.5 %**, ordered digests exact, prefill and decode unchanged. Read
together with the body above, the lever's payoff is conditional on the
P cluster's idle state: PerfHold's same-host A/B (co-tenant load, P
cluster never decayed) measured +0.3–2.7 % ttft rate and +1.0–2.3 %
prefill with zero regressions; w71's decayed-cluster run measured −8.7 %
first-token and +10.5 % ttft rate. The shipped default (`1024`) is
harmless in both states and pays in the decayed one.
