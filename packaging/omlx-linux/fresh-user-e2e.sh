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
# Set values to the ticket's exact wheel and the model already in that host's HF cache.
export HOME=/tmp/omlx-fresh-user
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
WHEEL=/path/to/compatible-mlx-omarchy.whl
MODEL=mlx-community--Qwen3-4B-Instruct-2507-4bit

# 1. Isolated fresh install (oMLX v0.7.0 / pinned compatibility patches).
cd ~/src/mlx-omarchy
python3 packaging/omlx-linux/install.sh --mlx-wheel "$WHEEL" --home "$HOME"

# 2. Start the real server; wait for its health check in another terminal.
"$HOME/.venvs/omlx/bin/omlx" serve --host 127.0.0.1 --port 8900 --model "$MODEL"

# 3. Verify server and model inventory (not sufficient by itself for feature acceptance).
curl --fail --max-time 10 http://127.0.0.1:8900/health
curl --fail --max-time 10 http://127.0.0.1:8900/v1/models

# 4. In a browser on this host, open http://127.0.0.1:8900/admin.
# Log in; load $MODEL from Models; send a prompt in Chat and save visible generated text.
# Capture dashboard/chat/benchmark screenshots at 375, 768, 1024, and 1440 CSS px widths.
# Open Benchmark, run PP/TG, save results, and verify no cached-token warning in its output/server log.
# The panel is intentionally no-cache at this pin; partial-prefix cache behavior is a separate test.
# Record real host/backend/model/version/timestamp and preserve raw result JSON/logs.

# 5. Stop only this server process with Ctrl-C. Preserve the run directory before cleanup.
EOF
