#!/usr/bin/env bash
# silero VAD per-frame latency on CPU, and the pre-filter's
# effect. Prints one RESULT line per number.
set -uo pipefail
PY="${VAD_PY:-$HOME/.local/share/mlx-omarchy/venv/bin/python}"
[ -x "$PY" ] || PY=python3
"$PY" - <<'EOF'
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("RESULT onnxruntime MISSING")
    raise SystemExit(0)

opts = ort.SessionOptions()
opts.inter_op_num_threads = 1
opts.intra_op_num_threads = 1
sess = ort.InferenceSession("silero_vad.onnx", sess_options=opts,
                            providers=["CPUExecutionProvider"])
state = np.zeros((2, 1, 128), dtype=np.float32)
sr = np.array(16000, dtype=np.int64)
x = np.zeros((1, 576), dtype=np.float32)  # 64-sample context + 512-sample frame
for _ in range(20):  # warm
    sess.run(None, {"input": x, "state": state, "sr": sr})
n = 500
t0 = time.perf_counter()
for _ in range(n):
    sess.run(None, {"input": x, "state": state, "sr": sr})
per_frame = (time.perf_counter() - t0) / n * 1000
print(f"RESULT per_32ms_frame_ms {per_frame:.3f}")
print(f"RESULT per_30ms_audio_ms {per_frame * 30 / 32:.3f}")
EOF
