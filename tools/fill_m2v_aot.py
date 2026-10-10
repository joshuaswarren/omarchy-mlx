#!/usr/bin/env python3
"""Fill the ahead-of-time metal2vk module cache for the shipped kernel set.

Runs m2v-compile over the assembled MSL files of the verified custom kernels
(the parity-sweep inputs) and writes one content-addressed module pair per
kernel into the AOT directory the wheel installs:

    <out>/<sha256(msl bytes + NUL + entry name)>.spv
    <out>/<sha256(msl bytes + NUL + entry name)>.json

The key is exactly the one mlx::core::omarchy::m2v::aot_cache_key computes at
dispatch time. Run this where the metal2vk toolchain is installed (CLANG, OPT,
CLSPV, SPIRV_VAL in the environment), before building a release wheel, so the
shipped AOT directory answers every verified kernel without a compiler:

    python3 tools/fill_m2v_aot.py \
        --msl-dir DIR1 --msl-dir DIR2 --out overlay/mlx/backend/omarchy/m2v_aot

The gate table inside m2v_route.cpp (between the M2V_GATE_TABLE markers) must
carry every kernel that compiles here; the script checks that and refuses to
write a manifest for an unmapped kernel.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTE_CPP = (
    REPO_ROOT / "overlay" / "mlx" / "backend" / "omarchy" / "m2v_route.cpp"
)


def entry_name(msl_text: str) -> str:
    """The m2v-compile entry: the host_name instantiation, else the single
    [[kernel]] function name. Mirrors m2v::kernel_entry_name in C++."""
    marker = '[[host_name("'
    at = msl_text.find(marker)
    if at >= 0:
        begin = at + len(marker)
        end = msl_text.find('"', begin)
        if end > begin:
            return msl_text[begin:end]
    at = msl_text.find("[[kernel]]")
    if at < 0:
        raise SystemExit(f"no kernel entry in an input file")
    declaration = msl_text.find("void ", at)
    if declaration < 0:
        raise SystemExit("no kernel function name in an input file")
    match = re.match(r"[A-Za-z0-9_]+", msl_text[declaration + 5 :])
    if not match:
        raise SystemExit("empty kernel function name in an input file")
    return match.group(0)


def base_name(entry: str) -> str:
    name = entry
    prefix = "custom_kernel_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    cut = name.find("__")
    return name[:cut] if cut >= 0 else name


def gate_table() -> dict:
    text = ROUTE_CPP.read_text()
    begin = text.index("M2V_GATE_TABLE_BEGIN")
    end = text.index("M2V_GATE_TABLE_END")
    block = text[text.index("{", begin) : text.rindex("}", begin, end) + 1]
    return json.loads(block)["kernels"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--msl-dir",
        action="append",
        required=True,
        help="directory holding assembled .metal files (repeatable)",
    )
    parser.add_argument(
        "--out", required=True, help="AOT directory to fill (created)"
    )
    parser.add_argument(
        "--tool",
        default=os.environ.get("M2V_COMPILE", "m2v-compile"),
        help="m2v-compile path (default M2V_COMPILE or PATH)",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify the AOT directory already covers every kernel",
    )
    args = parser.parse_args()

    table = gate_table()
    inputs = sorted(
        pathlib.Path(directory) / path
        for directory in args.msl_dir
        for path in os.listdir(directory)
        if path.endswith(".metal")
    )
    if not inputs:
        raise SystemExit("no .metal inputs found")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    problems = []
    manifest_path = out_dir / "manifest.json"
    if args.check:
        if not manifest_path.exists():
            raise SystemExit(
                f"no manifest in {out_dir}; run the fill first"
            )
        manifest = json.loads(manifest_path.read_text())
        for path in inputs:
            msl = path.read_bytes()
            name = entry_name(msl.decode())
            base = base_name(name)
            key = hashlib.sha256(msl + b"\0" + name.encode()).hexdigest()
            state = manifest.get(base, {}).get("state")
            has_module = (out_dir / f"{key}.spv").exists() and (
                out_dir / f"{key}.json"
            ).exists()
            if state in ("refused", "invalid"):
                if has_module:
                    problems.append(
                        f"{name} is {state} but a module is present"
                    )
                elif table.get(base, {}).get("state") != "failed":
                    problems.append(
                        f"{name} is recorded {state} but the gate table "
                        f"no longer marks {base} failed; regenerate the cache"
                    )
                continue
            if not has_module:
                problems.append(f"missing AOT module for {name} ({key})")
                continue
            if base not in table:
                problems.append(f"{name} has no gate table entry for {base}")
                continue
            manifest[base] = {"key": key, "entry": name, "base": base}
    else:
        for path in inputs:
            msl = path.read_bytes()
            name = entry_name(msl.decode())
            base = base_name(name)
            key = hashlib.sha256(msl + b"\0" + name.encode()).hexdigest()
            with tempfile.TemporaryDirectory(prefix="m2v-aot-") as work:
                result = subprocess.run(
                    [
                        args.tool,
                        "--msl",
                        str(path),
                        "--out",
                        work,
                        "--name",
                        name,
                        "--timeout",
                        str(args.timeout),
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=args.timeout * 4 + 60,
                    check=False,
                )
                if result.returncode == 0:
                    if base not in table:
                        problems.append(
                            f"{name} compiles but the gate table has no entry "
                            f"for {base}; add it to m2v_route.cpp"
                        )
                        continue
                    line = json.loads(result.stdout.strip().splitlines()[-1])
                    spv_path = pathlib.Path(line["spv"])
                    json_path = pathlib.Path(line["json"])
                    if not spv_path.is_absolute():
                        spv_path = pathlib.Path(work) / spv_path
                    if not json_path.is_absolute():
                        json_path = pathlib.Path(work) / json_path
                    shutil.copyfile(spv_path, out_dir / f"{key}.spv")
                    shutil.copyfile(json_path, out_dir / f"{key}.json")
                    manifest[base] = {
                        "key": key,
                        "entry": name,
                        "base": base,
                        "state": table[base]["state"],
                    }
                    print(f"compiled {name} -> {key[:12]}")
                elif result.returncode in (3, 5):
                    expected = base in table and table[base]["state"] == "failed"
                    kind = "refused" if result.returncode == 3 else "invalid"
                    if expected:
                        print(f"{kind:8s} {name} (expected: failed in the table)")
                        manifest[base] = {
                            "key": key,
                            "entry": name,
                            "base": base,
                            "state": kind,
                        }
                    else:
                        problems.append(
                            f"{name}: {kind} by metal2vk but the gate table "
                            f"does not mark it failed"
                        )
                else:
                    diagnostic = (result.stderr or result.stdout).strip()
                    first = diagnostic.splitlines()[0] if diagnostic else ""
                    problems.append(
                        f"{name}: m2v-compile exit {result.returncode}: {first}"
                    )

    if not args.check:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n"
        )
    verified = sum(
        1 for row in manifest.values() if row.get("state") == "verified"
    )
    print(
        f"{len(manifest)} kernels in the manifest, {verified} verified, "
        f"{len(problems)} problems"
    )
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
