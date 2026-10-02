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
