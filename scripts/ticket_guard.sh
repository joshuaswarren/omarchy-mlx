# ticket_guard.sh — source me from PairGates ticket scripts.
#
# Leak-free assistant lifecycle for shared-GPU tickets:
#   guard_kill_leftovers <home-fragment>...  kill MY leftover assistant
#       trees (match home path in cmdline) BEFORE the fuser gate, so a
#       previous ticket's SIGKILL'd wrapper cannot poison the next run.
#   guard_start_watchdog <home-fragment>...  start a setsid watchdog
#       (own session, survives SIGKILL of the wrapper and of gpu-turn)
#       that kills the assistant trees when this ticket PID vanishes on
#       ANY exit path.  Capped at 2 h so a recycled PID cannot hold it
#       forever; the next ticket's guard_kill_leftovers covers the rest.

guard_kill_leftovers() {
    for pat in "$@"; do
        pkill -KILL -f "$pat" 2>/dev/null || true
    done
    sleep 2
}

guard_start_watchdog() {
    local ticket=$$
    setsid nohup bash -c '
        ticket=$1; shift
        deadline=$(( $(date +%s) + 7200 ))
        while [ "$(date +%s)" -lt "$deadline" ]; do
            kill -0 "$ticket" 2>/dev/null || break
            sleep 5
        done
        kill -0 "$ticket" 2>/dev/null && exit 0  # PID recycled; ticket may live
        for pat in "$@"; do pkill -KILL -f "$pat" 2>/dev/null; done
    ' pairgates-watchdog "$ticket" "$@" >/dev/null 2>&1 &
    disown 2>/dev/null || true
}
