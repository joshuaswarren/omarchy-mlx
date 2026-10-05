#!/usr/bin/env bash
# FamGlmDsa M2 correctness ticket (<= 20 min): parity grid on real Apple
# GPU. Runs from a private copy of the shared omarchy venv with this
# branch's test files overlaid. No timing claims here (correctness only
# per the window policy); wall times are logged for later comparison
# but the deliverable is the parity table.
set -u
HOME_DIR="${HOME:?HOME must be explicit}"
RUN_DIR="$HOME_DIR/dsa-fam-correctness"
PY="$RUN_DIR/venv/bin/python"
LOG="$RUN_DIR/result.log"
export MLX_OMARCHY_WORK_DIR="$RUN_DIR/.work"

{
  echo "== FamGlmDsa M2 correctness run =="
  echo "host: $(hostname)"
  echo "date -u: $(date -u)"
  echo "python: $PY"
  "$PY" -c 'import mlx.core as mx; print("mlx:", mx.__version__ if hasattr(mx, "__version__") else "n/a"); print("metal.is_available():", mx.metal.is_available()); print("default_device:", mx.default_device())'
  echo "== prefill parity grid =="
  cd "$RUN_DIR/tests/dsa_indexer"
  "$PY" parity_speed.py --out parity_speed_m2.csv --n-warmup 3 --n-trials 5
  echo "== large-shape parity (hang on lavapipe; real GPU here) =="
  "$PY" large_shape_parity.py --out large_parity_m2.csv
  echo "== decode parity grid =="
  "$PY" decode_parity_speed.py --out decode_parity_m2.csv
  echo "== sparse MLA smoke =="
  "$PY" test_sparse_mla.py
  echo "== done =="
  echo "date -u: $(date -u)"
} > "$LOG" 2>&1
echo "log at $LOG"
tail -5 "$LOG"
