#!/bin/bash
# run_prune_probe.sh: 4 CPU ranks, prune off then on, same checkpoint. Logs in /tmp/w6z-prune{0,1}.log, summary on stdout.
cd "$HOME/scratch/w6z/muse" || exit 1
python3 - <<'EOF'
p = "prune_probe.py"
s = open(p).read()
if "del m, flat" not in s:
    s = s.replace('    print("rank 0 wrote checkpoint", flush=True)\n', '    print("rank 0 wrote checkpoint", flush=True)\n    del m, flat, out, v\n    gc.collect()\n')
    open(p, "w").write(s)
EOF
rm -f /tmp/w6z-prune.safetensors
for P in 0 1; do
  timeout 100 venv032/bin/mlx.launch --backend ring --hosts 127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1 \
    --env PYTHONUNBUFFERED=1 --env MLX_OMARCHY_PIPELINE_PRUNE=$P venv032/bin/python prune_probe.py /tmp/w6z-prune.safetensors > /tmp/w6z-prune$P.log 2>&1
  echo "== PRUNE=$P"
  grep -E "^rank|Error|Traceback" /tmp/w6z-prune$P.log | sort | cut -c1-230
done
