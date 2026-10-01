#!/usr/bin/env bash
# gpu-turn wrapper that uses run_pair_direct-style launcher but goes
# through the assistant. Spawns the assistant, runs the harness, kills
# the whole process group on ticket expiry. NOT yet used — kept for the
# next iteration if the coordinator path stays too slow.
set -uo pipefail
exec "$@"