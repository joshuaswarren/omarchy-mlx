"""Write the GPU STT acceptance receipt from measured results; refuses on any failure.

qualify.py <home> <scores.json> <latency.json> <worker.gdb.txt> <source>
"""
import json, sys
from pathlib import Path

from mlx_omarchy_assistant import gpu_stt

home, scores, latency, gdb_log, source = sys.argv[1:6]
parakeet = json.loads(Path(scores).read_text())["parakeet_gpu"]
lat = json.loads(Path(latency).read_text())
count_line = next(l for l in Path(gdb_log).read_text().splitlines() if l.startswith("CPU_COUNT "))
count = json.loads(count_line[len("CPU_COUNT "):])
if not count["resolved_while_running"]:
    sys.exit(f"refused: CPU counter breakpoint never resolved ({count})")
receipt = gpu_stt.write_acceptance_receipt(
    home,
    wer_by_subset={k: parakeet[k]["wer"] for k in ("test-clean", "test-other", "accented", "mixed_0dB")},
    empty_rate_by_subset={k: parakeet[k]["empty_rate"] for k in ("silence", "noise")},
    latency_ms={"p50": lat["p50_ms"], "p95": lat["p95_ms"]},
    cpu_tensor_events=count["cpu_command_encoder_calls"],
    receipt_source=source, host="<project-m2> (M2 Max T6021)")
print(json.dumps(receipt, indent=1))
