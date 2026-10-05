#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Driver pass: run every extracted metal_kernel site through the
dev-box translator driver and emit CSV v3.

Synthetic signature: inputs become `const device float*`, outputs
`device float*` — construct verdicts are exact; dtype-sensitive paths
(bf16/int8 branches) are refined per-kernel when the kernel runs.
"""
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kernel_battery import extract_from_tree  # noqa: E402

DRIVER = "/tmp/translator_driver"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kernel-battery-static-v3.csv")


def wrap(name: str, source: str, input_names, output_names) -> str:
    args = [f"const device float* {n} [[buffer({i})]]"
            for i, n in enumerate(input_names or ["in"])]
    base = len(args)
    args += [f"device float* {n} [[buffer({base + i})]]"
             for i, n in enumerate(output_names or ["out"])]
    return (f"[[kernel]] void {name or 'probe'}(\n    "
            + ",\n    ".join(args) + ") {\n" + source + "\n}\n")


def main() -> None:
    rows = []
    for root, tag in ((Path("/tmp/omlx-tf-parity/omlx/omlx"), "omlx"),
                      (Path("/tmp/omlx-tf-parity/TensorFold/src"), "tf")):
        for k in extract_from_tree(root, tag):
            wrapped = wrap(k.name, k.msllib, k.input_names or None,
                           k.output_names or None)
            with tempfile.NamedTemporaryFile("w", suffix=".msl",
                                             delete=False) as f:
                f.write(wrapped)
                path = f.name
            try:
                proc = subprocess.run([DRIVER, path], capture_output=True,
                                      text=True, timeout=30)
                verdict = proc.stdout.splitlines()[0] if proc.stdout else ""
                err = ""
                if "reject" in verdict:
                    for line in proc.stdout.splitlines():
                        if line.startswith("ERR "):
                            err = line[4:]
                status = "accept" if "accept" in verdict else "reject"
            except subprocess.TimeoutExpired:
                status, err = "timeout", ""
            finally:
                os.unlink(path)
            rows.append((tag, Path(k.file).name, k.name or "<anon>", k.line,
                         status, err, "|".join(k.gmsl_features),
                         len(k.msllib)))
    with OUT.open("w") as fh:
        fh.write("source,file,name,line,ttf_status,ttf_error,gmsl,msl_size\n")
        for r in rows:
            fh.write(",".join(str(x).replace(",", ";") for x in r) + "\n")
    from collections import Counter
    c = Counter(r[4] for r in rows)
    print("total:", len(rows), dict(c))
    for r in rows:
        if r[4] != "accept":
            print(f"  {r[0]}:{r[1]}:{r[2]} [{r[4]}] {r[5][:90]}")


if __name__ == "__main__":
    main()
