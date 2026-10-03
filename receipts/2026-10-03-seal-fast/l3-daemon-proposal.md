# Proposal: make the persistent ANE worker daemon (L3) the CLI default, with an idle timeout

Status: PROPOSAL — not implemented. Evidence base:
`receipts/2026-10-02-parakeet-warm-clip/` (L3 measured on the M1: fresh
`transcribe` per-call wall 1938.8 ms private vs **852.7 ms** daemon-attached,
interleaved A/B n=10 per arm, idle-gated), plus the SealFast seal findings
(`README.md` in this directory): the seal is read+hash-bound at the ARMv8
crypto floor (312 ms, copy already fully overlapped), and the root-chain
stamp removes it only for system installs. The daemon removes the whole
remaining session-open path (~800-950 ms) for EVERY install and is the
larger, general lever.

## Design (as measured, minus the opt-in)

`mlx-omarchy-ane-worker --daemon --socket PATH --idle-time-ms N` already
exists behind `MLX_OMARCHY_PK_KEEP_WORKER=1` + `MLX_OMARCHY_ANE_SOCK=PATH`:
it opens the same sealed resident session (libane pin and bundle hashes
checked at daemon start), binds `PATH` mode 0600, refuses cross-uid peers
(SO_PEERCRED), holds a single-instance `flock` on `PATH.lock`, serves one
client at a time, and exits on the idle timer, SIGTERM/SIGINT, `--stop`, or
a lost session. The client attaches with a 1 s connect bound and falls back
to a private worker on any connect failure.

Change: default `KEEP_WORKER` on, socket at
`${XDG_RUNTIME_DIR:-/run/user/$uid}/mlx-omarchy/ane-worker-$version.sock`,
idle timeout 5 min (env override `MLX_OMARCHY_PK_WORKER_IDLE_MS`), kill
switch `MLX_OMARCHY_PK_KEEP_WORKER=0`.

## Risks and mitigations

1. **Device ownership.** Today each private worker holds the ANE device fd
   only for its process lifetime and lanes serialize via per-run locks
   (`/tmp/m1-gpu.lock`, `/var/tmp/ane-run.lock`, gpuwin windows). A
   resident daemon holds the device across invocations, so a lane's
   "lock per run" no longer bounds device tenure. Mitigation: the daemon
   must acquire the same per-run lock (or a shared variant) around each
   batch submit, not hold it for its lifetime; single-client-at-a-time
   serialization stays. This needs an explicit fleet-lane review before
   default-on — it changes contention behavior for every other lane.
2. **Stale daemon after package upgrade.** The daemon pins and seals
   libane + bundles at start; a pacman/venv upgrade swaps the files and a
   surviving daemon serves the OLD bytes. Integrity is intact (sealed
   bytes are self-consistent) but the CLI and daemon would disagree about
   versions. Mitigation: socket name carries the wheel build stamp
   (`$version.sock`), and the client verifies the daemon's reported
   stamp == its own before attach; any mismatch → `--stop` the stale
   daemon and spawn a private worker. Package upgrade hooks (post-install)
   may also `--stop` any running daemon for the replaced version.
3. **Unix socket security.** Mode 0600 + SO_PEERCRED same-uid already
   block other users. Residual risks: a predictable socket path enables a
   pre-bind symlink squat by the SAME user only (mitigated by
   XDG_RUNTIME_DIR, root-owned per-user, mode 0700); a malicious same-uid
   process is out of threat model (it already owns the user's files).
   Stale socket files after a crash are handled by the existing
   connect-fails → private-worker fallback.
4. **Idle timeout.** Too short: the ~800 ms session open returns every
   idle-gap (the cost we are removing). Too long: the daemon keeps the
   sealed program + weights resident (hundreds of MB resident) and holds
   the device capability open. 5 min default covers a batch of clips;
   the idle timer resets on each submit; SIGTERM/systemd user unit gives
   deterministic teardown; the kill switch restores today's behavior
   exactly.
5. **Crash isolation.** A daemon crash must not corrupt outputs: the
   client already treats a lost session as a hard error for that pass and
   falls back to a private worker for the next; the worker's fail-safe
   refusal paths are unchanged.

## Acceptance before default-on

- Paired A/B (daemon default vs private default) on the M1 and M1 Max:
  per-call wall, session open, golden contract on every cell, idle-gated.
- Concurrent-lane drill: two lanes using daemons plus one lane using
  per-run locks — no deadlock, no starvation, lock audit clean.
- Upgrade drill: wheel upgrade with a live daemon → stale-daemon
  detection, `--stop`, and a clean private-worker run.
