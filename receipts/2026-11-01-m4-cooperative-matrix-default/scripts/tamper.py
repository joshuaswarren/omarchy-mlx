"""Write a copy of a bits dump with one 16-bit word flipped in case c0
(the mismatching baseline for the compare exit-status check).
usage: tamper.py <in.npz> <out.npz>"""
import sys

import numpy as np

d = dict(np.load(sys.argv[1]))
a = d["c0"].copy().reshape(-1)
a[0] ^= np.array(1, dtype=a.dtype)
d["c0"] = a.reshape(d["c0"].shape)
np.savez(sys.argv[2], **d)
print("tampered c0[0]")
