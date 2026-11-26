#!/usr/bin/env bash
# Command list for the fresh-user oMLX path. Hardware commands run only after a gpu-turn ticket.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fresh-user-e2e.sh --list

Prints the pre-registered install/server/UI/benchmark sequence for a real omarchy-mlx host.
This script intentionally does not install packages, start a server, or contact hardware.
Run each printed command inside a gpu-turn ticket on the assigned GPU host; use the browser
steps to exercise the actual admin UI. The dev-box must not be used for model or benchmark data.
EOF
}

if [[ ${1:-} != --list ]]; then
  usage >&2
  exit 2
fi

cat <<'EOF'
# Run inside the gpu-turn ticket on the assigned Omarchy M+ GPU host.
# Start from the real user's shell so cached weights remain available to the fresh install.
HOST_HOME="${HOME:?run from the assigned user's login environment}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOST_HOME/.cache/huggingface/hub}"
export HOME=/tmp/omlx-fresh-user
WHEEL=/path/to/compatible-mlx-omarchy.whl
MODEL=mlx-community--Qwen3-4B-Instruct-2507-4bit

# 1. Isolated fresh install (oMLX v0.7.0 / pinned compatibility patches).
cd ~/src/mlx-omarchy
python3 packaging/omlx-linux/install.sh --mlx-wheel "$WHEEL" --home "$HOME"

# 2. Start without --model so the admin UI performs the model-load action.
"$HOME/.venvs/omlx/bin/omlx" serve --host 127.0.0.1 --port 8900

# 3. In another terminal, verify the server and discovered model inventory.
curl --fail --max-time 10 http://127.0.0.1:8900/health
curl --fail --max-time 10 http://127.0.0.1:8900/v1/models

# 4. In a browser on this host, open http://127.0.0.1:8900/admin.
# If prompted, log in using the configured API key; otherwise follow the loopback dashboard redirect.
# Load $MODEL from Models; wait for the loaded state; send a prompt in Chat and save generated text.
# Capture dashboard/chat/benchmark screenshots at 375, 768, 1024, and 1440 CSS px widths.
# Open Benchmark, run PP/TG, save the results page and server log, and verify the cached-token warning
# does not appear. The pinned panel is intentionally no-cache; partial-prefix cache is a separate test.
# Record host/backend/model/version/timestamps and preserve raw result JSON and logs.

# 5. Stop only this server process with Ctrl-C. Preserve the run directory before cleanup.
EOF
