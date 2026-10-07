#!/usr/bin/env python3
"""check-compute-kernel-wiring: static consistency gate for ComputeKernel.

Commit 5f05b1ee2 ("Bonsai family") landed shader dispatch rows that
referenced 17 ComputeKernel::Bonsai* values the overlay compute.h never
declared, so the feature never compiled. Nothing caught it because the
wiring spans four places with no cross-check:

  overlay/.../compute.h   enum class ComputeKernel { ... };
  overlay/.../compute.cpp ShaderBytes shader_bytes(ComputeKernel) switch
  overlay/.../CMakeLists  omarchy_shader(<name> ...) embeds SPIR-V blobs
  overlay/**              ComputeKernel::<Value> dispatch references

This gate re-derives those sets from the tree and fails when:

  1. a referenced ComputeKernel::<Value> is not declared in compute.h;
  2. a shader_bytes() row returns a symbol with no omarchy_shader() target
     (missing shader/SPIR-V build line);
  3. an omarchy_shader() target is never returned by any row (counts drift);
  4. a referenced value has no dispatch row and is not a sentinel (Custom,
     Count): create_pipeline() would throw invalid_argument at runtime;
  5. a target name ending in _r<N> lacks -DROWS_PER_SLOT=<N> on its build
     line (spec-constant/name drift).

INFO only (exit 0): declared values with no dispatch row and no reference.
These are dead entries kept for the append-only GPU-profile id convention;
the gate names them so the pile stays visible.

Pure stdlib text scan; no compiler, no network. Runtime is well under a
second on the full tree.

Usage:
  check_compute_kernel_wiring.py [repo_root]     (default: scripts/..)
Exit: 0 consistent, 1 inconsistent / unreadable inputs (fail-closed).
"""
import re
import sys
from pathlib import Path

SENTINELS = {"Custom", "Count"}
SOURCE_SUFFIXES = {".cpp", ".cc", ".cu", ".h", ".hpp"}

ENUM_RE = re.compile(r"enum class ComputeKernel : uint16_t \{(.*?)\n\};", re.S)
ENUM_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*(?:=[^,]+)?,", re.M)
REF_RE = re.compile(r"ComputeKernel::([A-Za-z_]\w*)")
SWITCH_HEAD_RE = re.compile(r"ShaderBytes shader_bytes\(ComputeKernel kernel\) \{")
CASE_RE = re.compile(r"case ComputeKernel::([A-Za-z_]\w*):")
RETURN_RE = re.compile(r"return \{\s*(\w+),")
SHADER_RE = re.compile(r"omarchy_shader\(\s*(\w+)\s+([^)]*)\)")
ROWS_PER_SLOT_RE = re.compile(r"_r(\d+)$")


def die(msg):
    sys.stderr.write("compute-kernel-wiring: %s\n" % msg)
    sys.exit(1)


def parse_declared(header_text):
    m = ENUM_RE.search(header_text)
    if not m:
        die("cannot find 'enum class ComputeKernel : uint16_t' in compute.h")
    return set(ENUM_VALUE_RE.findall(m.group(1)))


def parse_references(overlay_root):
    refs = {}
    for path in sorted(overlay_root.rglob("*")):
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        for lineno, line in enumerate(
            path.read_text(errors="replace").splitlines(), 1
        ):
            for value in REF_RE.findall(line):
                refs.setdefault(value, []).append("%s:%d" % (path, lineno))
    return refs


def parse_switch(cpp_text):
    """Return (covered: value -> symbol, break_only: set(value))."""
    m = SWITCH_HEAD_RE.search(cpp_text)
    if not m:
        die("cannot find 'ShaderBytes shader_bytes(ComputeKernel kernel)' "
            "in compute.cpp")
    depth = 0
    start = cpp_text.index("{", m.start())
    for i in range(start, len(cpp_text)):
        if cpp_text[i] == "{":
            depth += 1
        elif cpp_text[i] == "}":
            depth -= 1
            if depth == 0:
                body = cpp_text[start + 1:i]
                break
    else:
        die("shader_bytes switch body is unbalanced")
    body = re.sub(r"\s+", " ", body)
    covered, break_only = {}, set()
    segments = CASE_RE.split(body)  # [pre, label, seg, label, seg, ...]
    for label, segment in zip(segments[1::2], segments[2::2]):
        ret = RETURN_RE.search(segment)
        if ret:
            covered[label] = ret.group(1)
        elif "break;" in segment:
            break_only.add(label)
    return covered, break_only


def parse_shader_targets(cmake_text):
    targets = {}
    for name, defs in SHADER_RE.findall(cmake_text):
        if name in targets:
            die("duplicate omarchy_shader target '%s'" % name)
        targets[name] = defs
    return targets


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    omarchy = root / "overlay" / "mlx" / "backend" / "omarchy"
    for path in (omarchy / "compute.h", omarchy / "compute.cpp",
                 omarchy / "CMakeLists.txt"):
        if not path.is_file():
            die("missing %s" % path)
    declared = parse_declared((omarchy / "compute.h").read_text(errors="replace"))
    refs = parse_references(root / "overlay")
    covered, break_only = parse_switch((omarchy / "compute.cpp").read_text(errors="replace"))
    targets = parse_shader_targets((omarchy / "CMakeLists.txt").read_text(errors="replace"))

    failures = []

    undeclared = sorted(set(refs) - declared)
    if undeclared:
        failures.append(
            "referenced but not declared in compute.h:\n    " +
            "\n    ".join("%s (e.g. %s)" % (v, refs[v][0]) for v in undeclared))

    missing_shader = sorted({v for v, s in covered.items() if s not in targets})
    if missing_shader:
        failures.append(
            "dispatch rows with no omarchy_shader target:\n    " +
            "\n    ".join("%s -> %s" % (v, covered[v]) for v in missing_shader))

    unwired = sorted(set(targets) - set(covered.values()))
    if unwired:
        failures.append(
            "omarchy_shader targets never returned by a dispatch row:\n    " +
            "\n    ".join(unwired))

    unbacked = sorted(
        v for v in refs if v not in covered and v not in SENTINELS)
    if unbacked:
        failures.append(
            "referenced values with no shader_bytes dispatch row "
            "(create_pipeline would throw):\n    " +
            "\n    ".join("%s (e.g. %s)" % (v, refs[v][0]) for v in unbacked))

    bad_spec = []
    for name, defs in sorted(targets.items()):
        m = ROWS_PER_SLOT_RE.search(name)
        if m and "-DROWS_PER_SLOT=%s" % m.group(1) not in defs:
            bad_spec.append("%s: missing -DROWS_PER_SLOT=%s" % (name, m.group(1)))
    if bad_spec:
        failures.append(
            "_r<N> shader targets with wrong ROWS_PER_SLOT spec constant:\n    " +
            "\n    ".join(bad_spec))

    dead = sorted(declared - set(covered) - set(break_only) - set(refs))
    dead = [v for v in dead if v not in SENTINELS]

    if failures:
        sys.stderr.write(
            "compute-kernel-wiring: %d problem(s)\n  - %s\n"
            % (len(failures), "\n  - ".join(failures)))
        sys.exit(1)

    print(
        "compute-kernel-wiring: ok — %d declared, %d referenced, "
        "%d dispatch rows, %d shader targets, %d dead entries"
        % (len(declared), len(refs), len(covered), len(targets), len(dead)))
    for v in dead:
        print("compute-kernel-wiring: info: declared but unreferenced and "
              "unwired (profile-id placeholder): %s" % v)


if __name__ == "__main__":
    main()
