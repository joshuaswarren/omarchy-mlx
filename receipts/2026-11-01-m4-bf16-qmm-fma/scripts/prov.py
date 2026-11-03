"""Provenance gate shared by every probe in this directory.

Prints the scripts/mlx_provenance.py line for the loaded wheel and refuses
(exit 3) unless the RECORD hashes match and the wheel's version stamp
ends in +<EXPECT_COMMIT> (or +diag.<EXPECT_COMMIT>), the commit this side
of an A/B was meant to be.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import mlx_provenance  # noqa: E402


def check():
    prov = mlx_provenance.installed_provenance()
    line = mlx_provenance.provenance_line(prov)
    print(line, flush=True)
    want = os.environ.get("EXPECT_COMMIT", "")
    ver = prov.get("dist_version") or ""
    stamp = ver.rsplit("+", 1)[-1] if "+" in ver else ""
    ok_stamp = bool(want) and stamp in (want, f"diag.{want}")
    if prov.get("verified") != "match" or not ok_stamp:
        print(f"PROVENANCE REFUSED: verified={prov.get('verified')} stamp={stamp!r} expected={want!r}",
              flush=True)
        sys.exit(3)
    return prov
