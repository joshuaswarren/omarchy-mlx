#!/usr/bin/env bash
# SushiHang: finish the sushi build on jw16 and arm the repro capture.
set -euo pipefail
cd /var/tmp/sushi-rr

cat > finish-build.sh <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd /var/tmp/sushi-rr
bash scripts/fetch-zig.sh >> zigfetch.log 2>&1
ZIG=$(ls -d /var/tmp/sushi-rr/zig-aarch64-linux-*/zig 2>/dev/null | head -1)
echo "ZIG=$ZIG" >> sushibuild.log
"$ZIG" build -Doptimize=ReleaseFast >> sushibuild.log 2>&1
ls -la zig-out/bin/ >> sushibuild.log 2>&1
echo SUSHI-BUILD-DONE >> sushibuild.log
EOF
chmod +x finish-build.sh

cat > capture-hang.sh <<'EOF'
#!/usr/bin/env bash
# Run the rebuilt sushi serve briefly; if it parks, capture everything.
set -uo pipefail
BIN=/var/tmp/sushi-rr/zig-out/bin/sushi
PACK=/var/tmp/pack
LOG=/var/tmp/sushi-rr/repro-run.log
[ -x "$BIN" ] || { echo "sushi binary missing"; exit 1; }
"$BIN" --help >> "$LOG" 2>&1 || true
# The exact kld/serve invocation is reconstructed from --help output above.
SUSHI_NGRAM_WARM=0 MLX_OMARCHY_TRACE_DISPATCH=1 timeout -k 20 600 \
  "$BIN" serve --model "$PACK" >> "$LOG" 2>&1 &
PID=$!
sleep 90
ST=$(ps -o stat= -p $PID 2>/dev/null || true)
if [[ "$ST" == *D* || "$ST" == *S* ]]; then
  CPU=$(ps -o %cpu= -p $PID 2>/dev/null || echo 0)
  if python3 -c "import sys; sys.exit(0 if float('${CPU:-0}') < 1.0 else 1)" 2>/dev/null; then
    echo "PARKED pid=$PID cpu=$CPU — capturing" >> "$LOG"
    cat /proc/$PID/stack >> "$LOG" 2>/dev/null || true
    for t in /proc/$PID/task/*; do
      echo "== tid $(basename $t)" >> "$LOG"
      cat $t/stack >> "$LOG" 2>/dev/null || true
    done
    dmesg | tail -40 >> "$LOG" 2>/dev/null || true
    kill -9 $PID 2>/dev/null || true
  fi
fi
wait $PID 2>/dev/null || true
echo CAPTURE-DONE >> "$LOG"
EOF
chmod +x capture-hang.sh
nohup nice -n 10 bash finish-build.sh >> finishbuild.log 2>&1 &
echo chain-started
