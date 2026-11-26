"""Worst single-call numbers of the per-call diagnostic of the 2-bit fidelity run.

Reads the S and R npz files of the run (array `diag`, shape windows x calls x 4: max |err|, rms err, rms ref, number of bf16 outputs differing)
and prints, per arm, the worst single-call ratio rms err / rms ref, the worst max |err|, the largest share of a call's outputs that differ from the
float64 reference, and the mean rms err. The call has 8 experts x 4096 outputs = 32768 outputs.
usage: python3 per_call_worst.py S.npz R.npz
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

OUTPUTS = 8 * 4096

for path in sys.argv[1:]:
    z = np.load(path)
    d = z["diag"]
    ratio = d[:, :, 1] / d[:, :, 2]
    share = d[:, :, 3] / OUTPUTS * 100.0
    print(f"arm {z['arm']}  file sha256 {hashlib.sha256(Path(path).read_bytes()).hexdigest()}")
    print(f"  calls {ratio.size} (windows {d.shape[0]} x calls per window {d.shape[1]})")
    print(f"  worst single-call rms err / rms ref: {ratio.max():.4g}")
    print(f"  worst single-call max |err|: {d[:, :, 0].max():.4g}")
    print(f"  largest share of a call's outputs that differ from the float64 reference: {share.max():.4f} percent")
    print(f"  mean rms err over all calls: {d[:, :, 1].mean():.4g}")
