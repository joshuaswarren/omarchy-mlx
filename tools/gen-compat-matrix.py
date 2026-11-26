#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Generate the omarchy primitive compatibility matrix.

Reads the backend sources and the test suite, derives one row per
primitive implementation entry, and
writes a markdown document to stdout. Redirect into
docs/compatibility-matrix.md. `--json-out PATH` also writes the
coverage badge data in shields.io endpoint format (docs/coverage.json).
Coverage counts a primitive only when its eval path really dispatches
compute and a TEST_CASE verifies values against host references. An
eval body whose only reachable outcome is `omarchy::unsupported` is a
refusal; cases that only pin refusals, only assert no-error smoke, or
carry a loud SKIP marker do not anchor coverage. See parse_test_cases
and build_row.

Sources (read-only):
  overlay/mlx/backend/omarchy/{primitives,custom_kernel}.cpp  entries and gates
  overlay/mlx/backend/omarchy/compute.h       ComputeKernel variants
  overlay/mlx/backend/omarchy/copy.cpp        engine kernels behind copies
  overlay/tests/omarchy/**/*.cpp              TEST_CASE bodies (anchor evidence)
  .work/mlx/mlx/primitives.h                  upstream primitive classes
  .work/mlx/mlx/fast_primitives.h             upstream fast:: primitives
  .work/mlx/mlx/{,io/,distributed/}*.cpp      op -> primitive construction map
  .work/mlx/mlx/distributed/primitives.h      upstream distributed:: primitives
  .work/mlx/mlx/backend/gpu/primitives.cpp    shared GPU eval_gpu definitions
  .work/mlx/mlx/backend/metal/*.cpp           Mac-usable signal (eval_gpu)
  .work/mlx/mlx/backend/{cpu,common}/*.cpp    Mac-usable signal (eval_cpu)
"""

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PRIMITIVES = ROOT / "overlay/mlx/backend/omarchy/primitives.cpp"
CUSTOM_KERNEL = ROOT / "overlay/mlx/backend/omarchy/custom_kernel.cpp"
COMPUTE_H = ROOT / "overlay/mlx/backend/omarchy/compute.h"
COPY_CPP = ROOT / "overlay/mlx/backend/omarchy/copy.cpp"
TEST_GLOB = "overlay/tests/omarchy/**/*.cpp"

UPSTREAM_H = ROOT / ".work/mlx/mlx/primitives.h"
UPSTREAM_FAST_H = ROOT / ".work/mlx/mlx/fast_primitives.h"
UPSTREAM_DIST_H = ROOT / ".work/mlx/mlx/distributed/primitives.h"

# Classes in the upstream headers that are dispatch bases, not
# primitives themselves.
UPSTREAM_BASES = {"Primitive", "UnaryPrimitive", "Custom", "DistPrimitive"}

UPSTREAM_CLASS = re.compile(
    r"^class (?:MLX_API\s+)?(\w+) : public (\w+) \{", re.M)

# Helpers whose fragments are generic engine limits, not per-primitive
# constraints. They appear once in the footnote instead of every row.
GENERIC_HELPERS = {"checked_u32", "checked_item_offset"}

NAME_VARS = {"name", "tag", "operation_name", "error_name"}

# Test-side evidence. A case anchors coverage only by asserting
# values; refusal pins, no-error smoke checks, and loud SKIP markers
# do not. The suite expresses refusal pins by asserting on
# evaluation_error() results or on variables bound to them, so
# assertion classification tracks those variables through the body.
ASSERT_THROW = re.compile(r"\b(?:CHECK|REQUIRE)_THROWS\w*\s*\(")
ASSERT_PLAIN = re.compile(
    r"\b(?:CHECK|REQUIRE)"
    r"(?:_(?:FALSE|EQ|NE|GT|GE|LT|LE|UNARY_FALSE|UNARY|MESSAGE))?\s*\(")
DOCTEST_SKIP = re.compile(r"\bSKIP\s*\(")
SKIP_LITERAL = re.compile(r'"SKIP\b')
ERROR_VAR_DECL = re.compile(
    r"(?:const\s+)?(?:std::string|bool|auto)\s+(\w+)\s*=\s*([^;]+);")
CATCH_BLOCK = re.compile(r"catch\s*\(")
ASSIGNMENT = re.compile(r"\b(\w+)\s*=(?![=])")
UNSUPPORTED_CALL = re.compile(r"\bomarchy::unsupported\s*\(")
THROW_CALL = re.compile(r"\bthrow\s+std::\w*\s*\(")
MULTIRANK_GROUP = re.compile(
    r"group\.size\(\)\s*(?:,\s*2|==\s*2|>=\s*2|>\s*1|!=\s*1|!=\s*2)")

# Upstream short-circuits every distributed op at group.size() == 1
# (.work/mlx/mlx/distributed/ops.cpp), so a value assertion in a
# single-process run verifies the short-circuit, never the primitive. A
# value anchor for a distributed:: primitive is accepted only from a case
# whose body carries a group.size() proof (assertion or FAIL guard); the
# two-rank harness expresses its guard as FAIL control flow so the guard
# cannot read as a value assertion, keeping value credit for the data
# comparisons alone.
DISTRIBUTED_NS = "distributed"

DTYPE_LABELS = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "int32": "i32",
    "uint32": "u32",
    "int64": "i64",
    "bool_": "bool",
}

VARIANT_TOKENS = {
    "F32": "f32",
    "F16": "f16",
    "BF16": "bf16",
    "I32": "i32",
    "U32": "u32",
    "Bool": "bool",
}

# Upstream sources whose function bodies construct primitives. Anchor
# resolution maps every op call in a TEST_CASE body through these
# bodies (see parse_op_constructions), so op-to-primitive resolution
# is derived from the sources instead of a hand-maintained table.
CONSTRUCTION_GLOBS = [
    ".work/mlx/mlx/*.cpp",
    ".work/mlx/mlx/io/*.cpp",
    ".work/mlx/mlx/distributed/*.cpp",
]

DTYPE_ORDER = ["f32", "f16", "bf16", "i32", "u32", "i64", "bool"]

UPSTREAM_GPU_PRIMITIVES = ROOT / ".work/mlx/mlx/backend/gpu/primitives.cpp"
METAL_BACKEND = ROOT / ".work/mlx/mlx/backend/metal"
CPU_BACKEND = ROOT / ".work/mlx/mlx/backend/cpu"
COMMON_BACKEND = ROOT / ".work/mlx/mlx/backend/common"
UNCLOSABLE_ON_VULKAN = set()

QUALIFIED_EVAL = re.compile(r"void (?:[\w:]+::)?(\w+)::eval_(gpu|cpu)\s*\(")

# Upstream declaration sites, for citations when a primitive has no
# eval_gpu and no eval_cpu anywhere.
UPSTREAM_DECL = {}


def brace_body(text, open_brace):
    """Brace-balanced body text starting at the '{' at open_brace."""
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace:index + 1]
    return ""


def parse_backend_eval_impls(directory, kind):
    """Primitive names the backend defines eval_<kind> for.

    Returns (real, throws): real maps name to "file:line"; throws maps
    name to "file:line" of an eval whose body is one unconditional
    throw. A body that checks arguments first and throws conditionally
    is real: it computes for the inputs it accepts.
    """
    real, throws = {}, {}
    for path in sorted(directory.glob("*.cpp")):
        text = strip_comments(path.read_text())
        for match in QUALIFIED_EVAL.finditer(text):
            name, seen_kind = match.group(1), match.group(2)
            if seen_kind != kind or name in real:
                continue
            line = text[:match.start()].count("\n") + 1
            cite = f"{path.relative_to(ROOT)}:{line}"
            body = brace_body(text, text.index("{", match.end()))
            if body[1:].lstrip().startswith("throw"):
                throws.setdefault(name, cite)
            else:
                real[name] = cite
    return real, throws


METAL_DIST_CPP = ROOT / ".work/mlx/mlx/backend/metal/distributed.cpp"
DIST_OPS_CPP = ROOT / ".work/mlx/mlx/distributed/ops.cpp"


def parse_upstream_metal_refusals(names):
    """Upstream primitive names upstream's own Metal backend refuses.

    The evidence file is upstream's Metal distributed backend: a
    throw naming the primitive means Metal does not implement it
    either. A missing file or a file without throws yields an empty
    set and the matrix omits the annotation; the annotation can never
    change the covered count or the denominator.
    """
    try:
        text = strip_comments(METAL_DIST_CPP.read_text())
    except OSError:
        return set()
    if "throw" not in text:
        return set()
    return {name for name in names if re.search(rf"\b{name}\b", text)}


def parse_distributed_singleton_guard():
    """Whether upstream's op layer short-circuits at group.size()==1.

    True means the distributed primitive is never constructed in a
    single-rank run, so a backend eval_gpu for it is unreachable by
    upstream design.
    """
    try:
        text = strip_comments(DIST_OPS_CPP.read_text())
    except OSError:
        return False
    return bool(re.search(r"size\(\)\s*==\s*1", text))


def parse_shared_gpu_primitives():
    """Names upstream's shared backend/gpu layer gives an eval_gpu to.

    A primitive with no omarchy entry resolves through this shared layer,
    which drives the omarchy copy engine via copy_gpu / fill_gpu /
    reshape_gpu / concatenate_gpu. If the layer is missing an eval_gpu the
    omarchy build would not link, so presence here is the honest signal
    that upstream handles the primitive generically on this backend.
    """
    text = strip_comments(UPSTREAM_GPU_PRIMITIVES.read_text())
    return set(re.findall(r"void (\w+)::eval_gpu\(", text))




def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def balanced(text, open_index):
    """Return the text inside the parenthesis that opens at open_index."""
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index]
    return text[open_index + 1:]


def function_body(lines, start):
    """Return the brace-balanced body text of the function at lines[start]."""
    joined = "\n".join(lines[start:start + 200])
    open_index = joined.find("{")
    depth = 0
    for index in range(open_index, len(joined)):
        char = joined[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return joined[open_index + 1:index]
    return joined[open_index + 1:]


def git(*args):
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def mask_noncode(text):
    """Blank comments and string/char literal contents, keeping length.

    Offsets in the result address the original text, so structural
    parsing never trips over braces or parens inside literals, and
    line numbers survive. Quote characters themselves stay.
    """
    out = list(text)
    length = len(text)
    index = 0
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = length if end == -1 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end == -1 else end + 2
        elif text[index] in "\"'":
            quote = text[index]
            end = index + 1
            while end < length and text[end] != quote:
                end += 2 if text[end] == "\\" else 1
            end = min(end + 1, length)
            for offset in range(index + 1, end - 1):
                if out[offset] != "\n":
                    out[offset] = " "
            index = end
            continue
        else:
            index += 1
            continue
        for offset in range(index, end):
            if out[offset] != "\n":
                out[offset] = " "
        index = end
    return "".join(out)


def block_end(masked, open_index):
    """Index of the brace closing the block opened at open_index."""
    depth = 0
    for index in range(open_index, len(masked)):
        char = masked[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(masked) - 1


NAMESPACE_TAIL = re.compile(r"\bnamespace\s+([A-Za-z_][\w:]*)\s*$")
NAMESPACE_ANON = re.compile(r"\bnamespace\s*$")
SCOPE_KEYWORD = re.compile(r"\b(struct|class|enum|union)\b")
CALL_SITE = re.compile(r"([A-Za-z_]\w*)\s*\(")
MAKE_SHARED = re.compile(r"make_shared<([A-Za-z_][\w:]*)>\s*\(")


def function_name(header):
    """Name before the balanced trailing parameter list of a definition.

    The last ')' in a definition header closes the parameter list, and
    its matching '(' is found by scanning back, so parens inside return
    types and parameters (std::function<...>) cannot hijack the name.
    Member methods keep their class prefix (Add::vjp), so upstream rule
    methods never merge with the free op functions the tests call.
    """
    close = header.rfind(")")
    if close == -1:
        return None
    depth = 0
    for index in range(close, -1, -1):
        char = header[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                member = re.search(
                    r"([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)\s*$",
                    header[:index])
                if member:
                    return "::".join(member.groups())
                match = re.search(r"([A-Za-z_]\w*)\s*$", header[:index])
                return match.group(1) if match else None
    return None


MAKE_SHARED = re.compile(r"make_shared<([A-Za-z_][\w:]*)>\s*\(")

# Primitives live in mlx::core (no namespace in the matrix rows),
# mlx::core::fast, and mlx::core::distributed; the namespace walk only
# sees the tail, so the tail maps to the matrix namespace.
NAMESPACE_ALIAS = {"core": ""}


def next_block(masked, start, end):
    """First '{' outside any parenthesis in masked[start:end].

    A brace inside a parameter list (a default argument such as
    StreamOrDevice s = {}) is not the function body opener.
    """
    depth = 0
    for index in range(start, end):
        char = masked[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "{" and depth == 0:
            return index
    return -1


def attribute_functions(masked):
    """(namespace, function name, body) for every function-definition block."""
    out = []

    def walk(start, end, ns):
        index = start
        while index < end:
            brace = next_block(masked, index, end)
            if brace == -1:
                return
            header = masked[index:brace].rstrip()
            close = block_end(masked, brace)
            tail = NAMESPACE_TAIL.search(header)
            if tail:
                tail_ns = tail.group(1).rpartition("::")[2]
                walk(brace + 1, close, NAMESPACE_ALIAS.get(tail_ns, tail_ns))
            else:
                name = function_name(header)
                if name:
                    out.append((ns, name, masked[brace + 1:close]))
            index = close + 1

    walk(0, len(masked), "")
    return out


def parse_op_constructions(upstream_keys):
    """Map every upstream op function name to the primitives it builds.

    Derived from the upstream sources in two passes. Pass one
    attributes each function body — including helpers inside
    anonymous namespaces, attributed to their enclosing named
    namespace — its own std::make_shared<Primitive> types and the
    functions it calls; an unqualified primitive name built inside an
    inner namespace (random::bits building RandomBits) falls back to
    its unique upstream namespace. Pass two walks the name-level call
    graph three hops out, so ops that build primitives through
    helpers (all_sum -> the internal all_reduce, bitwise_and ->
    in_binary, compile -> compile_fuse, the linalg wrappers) still
    resolve, with no curated list anywhere.

    Also derived: the VJP rule map, forward primitive class -> the
    rule primitives its X::vjp member builds (fast.cpp RMSNorm::vjp
    building RMSNormVJP), and the tracing entry ops, op functions
    whose call graph reaches the free vjp machinery. The tape
    backward dispatches primitive->vjp virtually, a call shape the
    name-level graph skips, so rule primitives are otherwise
    unreachable from any test call; parse_test_cases joins both maps
    at the case level instead.
    """
    direct = {}
    calls = {}
    vjp_rules = {}
    for pattern in CONSTRUCTION_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            masked = mask_noncode(path.read_text())
            for ns, op, body in attribute_functions_anon(masked):
                key = (ns, op)
                prims = direct.setdefault(key, set())
                edges = calls.setdefault(key, set())
                for match in MAKE_SHARED.finditer(body):
                    prim_ns, _, prim = match.group(1).rpartition("::")
                    if prim_ns and prim_ns not in ("fast", "distributed"):
                        continue
                    constructed = (prim_ns or ns, prim)
                    if constructed not in upstream_keys and not prim_ns:
                        owners = {
                            owner for owner, up in upstream_keys
                            if up == prim}
                        if len(owners) == 1:
                            constructed = (owners.pop(), prim)
                    if constructed in upstream_keys:
                        prims.add(constructed)
                        if op.endswith("::vjp"):
                            vjp_rules.setdefault(
                                op.rpartition("::")[0], set()).add(
                                    constructed)
                for call in CALL_SITE.finditer(body):
                    qualifier = body[max(0, call.start() - 6):call.start()]
                    if (qualifier.endswith("std::")
                            or qualifier.endswith(".")
                            or qualifier.endswith("->")):
                        continue
                    edges.add(call.group(1))
    by_name = {}
    for key in direct:
        by_name.setdefault(key[1], []).append(key)
    op_map = {}
    trace_ops = set()
    for name in by_name:
        built = set()
        seen = set()
        frontier = {name}
        for _ in range(3):
            following = set()
            for entry in sorted(frontier):
                if entry in seen:
                    continue
                seen.add(entry)
                for key in by_name.get(entry, ()):
                    built |= direct[key]
                    following |= calls[key]
            frontier = following - seen
        for constructed in built:
            op_map.setdefault(name, set()).add(constructed)
        if "vjp" in seen:
            trace_ops.add(name)
    return op_map, vjp_rules, trace_ops


def attribute_functions_anon(masked):
    """attribute_functions, also recursing into anonymous namespaces.

    Test helpers live in `namespace { ... }` blocks, which the
    construction-map walk skips; helper classification needs them.
    """
    out = []

    def walk(start, end, ns):
        index = start
        while index < end:
            brace = next_block(masked, index, end)
            if brace == -1:
                return
            header = masked[index:brace].rstrip()
            close = block_end(masked, brace)
            tail = NAMESPACE_TAIL.search(header)
            if tail:
                tail_ns = tail.group(1).rpartition("::")[2]
                walk(brace + 1, close,
                     NAMESPACE_ALIAS.get(tail_ns, tail_ns))
            elif NAMESPACE_ANON.search(header):
                walk(brace + 1, close, ns)
            else:
                name = function_name(header)
                if name:
                    out.append((ns, name, masked[brace + 1:close]))
            index = close + 1

    walk(0, len(masked), "")
    return out


def error_variable_names(body):
    """Names bound to refusal text in one test body.

    A name is refusal-bound when it is assigned inside a catch block,
    initialized from evaluation_error(...), or initialized from an
    expression mentioning an already refusal-bound name (the fixpoint
    follows bool flags derived from error strings).
    """
    names = set()
    for match in CATCH_BLOCK.finditer(body):
        args = balanced(body, match.end() - 1)
        brace = body.find("{", match.end() + len(args))
        if brace == -1:
            continue
        close = block_end(body, brace)
        for assign in ASSIGNMENT.finditer(body[brace + 1:close]):
            names.add(assign.group(1))
    changed = True
    while changed:
        changed = False
        for match in ERROR_VAR_DECL.finditer(body):
            name, rhs = match.group(1), match.group(2)
            if name in names:
                continue
            if "evaluation_error" in rhs or any(
                    re.search(rf"\b{re.escape(v)}\b", rhs)
                    for v in names):
                names.add(name)
                changed = True
    return names


def classify_assertions(body, error_vars):
    """(has value assertion, has refusal or no-error pin) for a body.

    CHECK/REQUIRE family macros carry the evidence. A *_THROWS* form,
    an argument reading evaluation_error() or a refusal-bound
    variable, or an argument asserting on a caught message pins a
    refusal; any other CHECK/REQUIRE asserts a computed value. WARN_*
    and MESSAGE record without failing and are not evidence.
    """
    value = pin = False
    if ASSERT_THROW.search(body):
        pin = True
    for match in ASSERT_PLAIN.finditer(body):
        args = balanced(body, match.end() - 1)
        if "evaluation_error" in args or any(
                re.search(rf"\b{re.escape(v)}\b", args)
                for v in error_vars):
            pin = True
        else:
            value = True
    return value, pin


def value_verify_helpers(masked):
    """File-local helper names whose bodies compare values.

    A helper verifies values when any of its definitions runs a value
    assertion itself or calls another verifying helper (check_floats
    ends in CHECK...). Overloads accumulate: the linalg suite defines
    expect_close twice, and dropping the asserting overload to a
    same-name shim would hide the verification. Helpers that format,
    quantize on the host, or only return refusal strings never
    verify.
    """
    bodies = {}
    for ns, name, body in attribute_functions_anon(masked):
        if name not in ("TEST_CASE", "SUBCASE") and ns == "":
            bodies.setdefault(name, []).append(body)
    calls = {
        name: {
            hit for hit in (
                match.group(1)
                for body in bodies[name]
                for match in CALL_SITE.finditer(body))
            if hit in bodies
        }
        for name in bodies
    }
    memo = {}

    def verify(name, seen):
        if name in memo:
            return memo[name]
        if name in seen:
            return False
        seen.add(name)
        value = any(
            classify_assertions(
                body, error_variable_names(body))[0]
            for body in bodies[name])
        result = value or any(
            verify(child, seen) for child in sorted(calls[name]))
        memo[name] = result
        return result

    for name in sorted(bodies):
        verify(name, set())
    return {name for name, ok in memo.items() if ok}


LAMBDA_ARG = re.compile(r"\]\s*\([^()]*\)\s*(?:->\s*[\w:<>]+)?\{")


def mapped_calls(text, op_map):
    """Primitives the op calls in text construct.

    Same call-site filter as the case loop: std::-qualified host
    references and method calls are not op calls.
    """
    out = set()
    for call in CALL_SITE.finditer(text):
        qualifier = text[max(0, call.start() - 6):call.start()]
        if qualifier.endswith(("std::", ".", "->")):
            continue
        out |= op_map.get(call.group(1), set())
    return out


def traced_regions(body, trace_ops):
    """Body sub-ranges of the callables passed to tracing entry ops.

    A vjp/value_and_grad call traces exactly its callable argument,
    so only op calls inside that callable land on the tape. Two
    shapes occur in the suite: an inline lambda in the argument
    list, and a named lambda bound earlier with auto name =
    [...](...) { ... }. Any other callable (a free function, a
    functor) resolves to no region, and the case earns no rule
    attribution from it.
    """
    regions = []
    for match in CALL_SITE.finditer(body):
        if match.group(1) not in trace_ops:
            continue
        args = balanced(body, match.end() - 1)
        lam = LAMBDA_ARG.search(args)
        if lam:
            brace = lam.end() - 1
            close = block_end(args, brace)
            regions.append(args[brace + 1:close])
            continue
        ident = re.match(r"\s*([A-Za-z_]\w*)", args)
        if not ident:
            continue
        decl_end = None
        for decl in re.finditer(
                rf"\bauto\s+{re.escape(ident.group(1))}\s*=",
                body[:match.start()]):
            decl_end = decl.end()
        if decl_end is None:
            continue
        brace = next_block(body, decl_end, match.start())
        if brace == -1 or brace >= match.start():
            continue
        regions.append(body[brace + 1:block_end(body, brace)])
    return regions


def parse_test_cases(op_map, upstream_keys, upstream_names, vjp_rules,
                     trace_ops):
    """TEST_CASEs with the primitives each body provably exercises.

    Evidence is content-derived from the case body:
    - ops: primitives reached by a direct op call. A call is the
      maximal identifier before '('; std::-qualified host references
      and method calls do not count, other qualifiers such as fast::
      do.
    - exclusive: the subset of ops reached by calls whose whole
      construction map is one primitive, plus primitives the body
      builds itself with std::make_shared (the contract tests build
      primitives no public op constructs).
    - mentions: upstream class names inside body string literals, the
      named-error qualifiers the case expects.
    - value / pin / skipped: whether the body verifies values (its
      own assertions or file-local value helpers), only pins refusals
      or no-error smoke, or carries a loud SKIP marker. A SKIP marker
      withdraws value credit for the whole case; a value anchor also
      requires a real call path, never a bare string mention.
    - trace: a case that calls a tracing entry op (vjp,
      value_and_grad, ...) provably runs the tape backward for the
      op calls inside the callable it passes: the virtual
      primitive->vjp dispatch builds each forward primitive's VJP
      rule primitive there. traced_regions resolves that callable,
      so calls outside it (setup, composed references) earn no rule;
      the rule joins ops, never exclusive, since one trace can
      carry many forward primitives.
    """
    cases = []
    order = {}
    for path in sorted(ROOT.glob(TEST_GLOB)):
        raw = path.read_text()
        masked = mask_noncode(raw)
        rel = path.relative_to(ROOT)
        if rel not in order:
            order[rel] = len(order)
        helpers = value_verify_helpers(masked)
        for match in re.finditer(r"TEST_CASE\s*\(", masked):
            line = masked[:match.start()].count("\n") + 1
            title = re.match(
                r'TEST_CASE\s*\(\s*"((?:[^"\\]|\\.)*)"',
                raw[match.start():]).group(1)
            args = balanced(masked, match.end() - 1)
            brace = masked.find("{", match.end() + len(args) + 1)
            close = block_end(masked, brace)
            body = masked[brace + 1:close]
            raw_body = raw[brace + 1:close]
            ops = set()
            exclusive = set()
            for call in CALL_SITE.finditer(body):
                qualifier = body[max(0, call.start() - 6):call.start()]
                if (qualifier.endswith("std::")
                        or qualifier.endswith(".")
                        or qualifier.endswith("->")):
                    continue
                mapped = op_map.get(call.group(1), set())
                ops |= mapped
                if len(mapped) == 1:
                    exclusive |= mapped
            for built in MAKE_SHARED.finditer(raw_body):
                prim = built.group(1)
                matches = {
                    up_ns for up_ns, up_prim in upstream_keys
                    if up_prim == prim}
                if len(matches) == 1:
                    constructed = (next(iter(matches)), prim)
                    ops.add(constructed)
                    exclusive.add(constructed)
            tape = set()
            for region in traced_regions(body, trace_ops):
                tape |= mapped_calls(region, op_map)
            for _, prim in sorted(tape):
                ops |= vjp_rules.get(prim, set())
            literals = " ".join(
                re.findall(r'"((?:[^"\\]|\\.)*)"', raw_body))
            mentions = {
                name for name in upstream_names
                if re.search(rf"\b{re.escape(name)}\b", literals)}
            error_vars = error_variable_names(body)
            value, pin = classify_assertions(body, error_vars)
            skipped = bool(
                DOCTEST_SKIP.search(body)
                or SKIP_LITERAL.search(raw_body))
            if skipped:
                value = False
            if not value and not skipped:
                value = any(
                    hit.group(1) in helpers
                    for hit in CALL_SITE.finditer(body))
            cases.append({
                "order": order[rel],
                "rel": str(rel),
                "line": line,
                "name": title,
                "ops": ops,
                "exclusive": exclusive,
                "mentions": mentions,
                "value": value,
                "pin": pin,
                "skipped": skipped,
                "multirank": bool(MULTIRANK_GROUP.search(body)),
            })
    return cases


def anchor_for(display, cases):
    """(anchor link, anchor kind) for one primitive.

    Kind 'value': a case that reaches the primitive through a call
    (its own op, a composite op, or a direct std::make_shared) and
    asserts values. Kind 'pin': a case that reaches the primitive but
    only asserts refusal messages or no-error smoke. Tiers within a
    kind, strongest first: exclusive op, any constructing op, name in
    an expected-error string. Within a tier the earliest case in file
    order wins. A bare string mention can only anchor a pin.

    A distributed:: primitive anchors only from a case whose body
    proves a multi-rank group (see MULTIRANK_GROUP): at one rank
    upstream never constructs the primitive, so an unguarded case
    anchors nothing — not a value, and not a pin either.
    """
    ns, _, name = display.rpartition("::")
    key = (ns, name)
    best = {"value": None, "pin": None}
    for tier in ("exclusive", "ops", "mentions"):
        for case in cases:
            if ns == DISTRIBUTED_NS and not case["multirank"]:
                continue
            if tier == "mentions":
                hit = name in case["mentions"]
                kind = "pin"
                if not hit:
                    continue
            else:
                hit = key in case[tier]
                if not hit:
                    continue
                if case["value"]:
                    kind = "value"
                elif case["pin"]:
                    kind = "pin"
                else:
                    continue
            current = best[kind]
            if current is None or (case["order"], case["line"]) < (
                    current["order"], current["line"]):
                best[kind] = case
    if best["value"] is not None:
        case = best["value"]
        return (
            f"[`{case['name']}`](../{case['rel']}#L{case['line']})",
            "value")
    if best["pin"] is not None:
        case = best["pin"]
        return (
            f"[`{case['name']}`](../{case['rel']}#L{case['line']})"
            " pins refusal, no values",
            "pin")
    return "—", None


def parse_kernel_families():
    """Group compute.h ComputeKernel variants into families."""
    text = COMPUTE_H.read_text()
    body = re.search(
        r"enum class ComputeKernel[^{]*\{(.*?)\};", text, re.S).group(1)
    variants = re.findall(r"^\s*(\w+),\s*$", body, re.M)
    families = {}
    variant_map = {}
    for variant in variants:
        if variant == "Count":
            continue
        if variant.startswith("Cast"):
            base = "Cast"
        else:
            base = variant
            for suffix in ("BF16", "F32", "F16", "I32", "U32", "Bool"):
                if variant.endswith(suffix):
                    base = variant[: -len(suffix)]
                    break
        tokens = re.findall(r"BF16|F32|F16|I32|U32|Bool", variant)
        dtypes = {VARIANT_TOKENS[token] for token in tokens}
        entry = families.setdefault(
            base, {"variants": [], "dtypes": set()})
        entry["variants"].append(variant)
        entry["dtypes"] |= dtypes
        variant_map[variant] = dtypes
    return families, variant_map


def parse_capability_gates(primitives_text):
    """Extract the float16/bfloat16 gates from require_float_dtype."""
    text = strip_comments(primitives_text)
    match = re.search(
        r"void require_float_dtype\(.*?\{(.*?)\n\}", text, re.S)
    if not match:
        return []
    body = match.group(1)
    gates = []
    for dtype, first, second in re.findall(
            r"==\s*(float16|bfloat16)\s*&&\s*"
            r"\(!capabilities\.(\w+)\s*\|\|\s*!capabilities\.(\w+)\)", body):
        gates.append((dtype, first, second))
    return gates


def parse_helpers(primitives_text, variant_dtypes):
    """Map helper name to (clean body, unsupported fragments, dtypes)."""
    text = strip_comments(primitives_text)
    lines = primitives_text.splitlines()
    clean_lines = text.splitlines()
    helpers = {}
    pattern = re.compile(
        r"^(?:void|omarchy::ComputeKernel|uint32_t|bool|array)\s+"
        r"(dispatch_\w+|fill_broadcast_transport|require_float_dtype|"
        r"require_index_source_dtype|select_float_kernel|checked_u32|"
        r"checked_item_offset|materialize_batched_matrix)\(")
    for index, line in enumerate(clean_lines):
        match = pattern.match(line)
        if not match:
            continue
        body = strip_comments(function_body(lines, index))
        helpers[match.group(1)] = body
    fragments = {
        name: extract_fragments(body)
        for name, body in helpers.items()
    }
    called = {
        name: set(re.findall(
            r"\b(dispatch_\w+|fill_broadcast_transport|require_\w+|"
            r"select_float_kernel|checked_\w+|materialize_\w+)\(", body))
        for name, body in helpers.items()
    }
    dtypes = {}
    for name, body in helpers.items():
        found = set()
        for token, label in DTYPE_LABELS.items():
            if re.search(r"\b" + token + r"\b", body):
                found.add(label)
        for token in re.findall(r"omarchy::ComputeKernel::(\w+)", body):
            found |= variant_dtypes.get(token, set())
        dtypes[name] = found
    return helpers, fragments, called, dtypes


def render_expr(expr):
    """Render an unsupported() argument as a qualifier fragment."""
    literals = re.findall(r'"((?:[^"\\]|\\.)*)"', expr)
    identifiers = [
        token for token in re.findall(r"[A-Za-z_]\w*", expr)
        if token in NAME_VARS
    ]
    if literals and identifiers:
        text = "".join(literals).strip()
        if expr.lstrip().startswith('"'):
            return f"{text} {{name}}"
        return f"{{name}} {text}".strip()
    if literals:
        return "".join(literals).strip()
    return "{name}"


def extract_fragments(body):
    fragments = []
    for match in re.finditer(r"omarchy::unsupported\s*\(", body):
        args = balanced(body, match.end() - 1)
        depth = 0
        split = len(args)
        for index, char in enumerate(args):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                split = index
                break
        fragment = render_expr(args[:split])
        if fragment not in fragments:
            fragments.append(fragment)
    for match in re.finditer(r"\bthrow\s+std::\w+\s*\(", body):
        args = balanced(body, match.end() - 1)
        fragment = render_expr(args)
        if fragment not in fragments:
            fragments.append(fragment)
    return fragments


def kernel_reaching_helpers(helpers, called):
    """Helpers whose dispatch chain ends in a compute kernel launch.

    An eval body that only calls such helpers still computes; without
    this transitive check a body delegating to a dispatch helper
    outside HELPER_FAMILY would read as refusing everything.
    """
    direct = {
        name
        for name, body in helpers.items()
        if "omarchy::ComputeKernel::" in body
        or COPY_EVIDENCE.search(body)
    }
    reaching = set(direct)
    changed = True
    while changed:
        changed = False
        for name, children in called.items():
            if name in reaching:
                continue
            if any(child in reaching for child in children):
                reaching.add(name)
                changed = True
    return reaching


def helper_fragments(helper, fragments, called, seen=None):
    """Transitive named-error fragments of a dispatch helper."""
    if seen is None:
        seen = set()
    if helper in seen or helper in GENERIC_HELPERS:
        return []
    seen.add(helper)
    out = list(fragments.get(helper, []))
    for child in sorted(called.get(helper, ())):
        out.extend(helper_fragments(child, fragments, called, seen))
    return out


def helper_dtypes(helper, dtypes, called, seen=None):
    if seen is None:
        seen = set()
    if helper in seen:
        return set()
    seen.add(helper)
    out = set(dtypes.get(helper, ()))
    for child in called.get(helper, ()):
        out |= helper_dtypes(child, dtypes, called, seen)
    return out


def parse_primitives():
    """Parse primitive entries in source order."""
    texts = []
    entries = []
    index_of = {}
    for path in (PRIMITIVES, CUSTOM_KERNEL):
        text = path.read_text()
        texts.append(text)
        lines = text.splitlines()
        namespace = ""
        for number, line in enumerate(lines):
            match = re.match(
                r"^namespace (?:mlx::core::)?(fast|distributed) \{", line)
            if match:
                namespace = match.group(1)
            elif re.match(
                    r"^\} // namespace (?:mlx::core::)?(fast|distributed)\s*$",
                    line):
                namespace = ""
            macro = re.match(
                r"^OMARCHY_(UNSUPPORTED|UNSUPPORTED_MULTI|USE_FALLBACK|BINARY|"
                r"UNARY)\((\w+)", line)
            impl = re.match(r"^void (\w+)::eval_gpu\(", line)
            fallback = re.match(r"^bool (\w+)::use_fallback\(", line)
            if macro:
                kind, name = macro.group(1), macro.group(2)
                key = (namespace, name)
                entry = index_of.setdefault(
                    key, {"ns": namespace, "name": name, "kinds": [],
                          "order": len(index_of), "line": number + 1})
                entry["kinds"].append(kind)
            elif impl:
                name = impl.group(1)
                key = (namespace, name)
                entry = index_of.setdefault(
                    key, {"ns": namespace, "name": name, "kinds": [],
                          "order": len(index_of), "line": number + 1})
                entry["impl_body"] = function_body(lines, number)
                entry["impl_line"] = number + 1
            elif fallback:
                name = fallback.group(1)
                key = (namespace, name)
                entry = index_of.setdefault(
                    key, {"ns": namespace, "name": name, "kinds": [],
                          "order": len(index_of), "line": number + 1})
                entry["fallback_true"] = "return true" in function_body(
                    lines, number)
    entries = sorted(index_of.values(), key=lambda entry: entry["order"])
    return "\n".join(texts), entries


def parse_upstream_primitives():
    """Concrete primitive classes upstream MLX defines, as (ns, name).

    The coverage denominator is upstream's total primitive list, not
    the entries the omarchy backend happens to enumerate.
    """
    out = []
    for path, ns in [
        (UPSTREAM_H, ""),
        (UPSTREAM_FAST_H, "fast"),
        (UPSTREAM_DIST_H, "distributed"),
    ]:
        try:
            text = path.read_text()
        except OSError as exc:
            raise SystemExit(
                f"cannot read upstream header {path}: {exc}\n"
                "The coverage denominator needs the upstream MLX sources "
                "under .work/mlx; sync them and rerun.")
        for match in UPSTREAM_CLASS.finditer(text):
            name, base = match.groups()
            if name in UPSTREAM_BASES:
                continue
            if base not in UPSTREAM_BASES:
                raise SystemExit(
                    f"unrecognized base class {base!r} for upstream "
                    f"primitive {name!r} in {path}; extend UPSTREAM_BASES "
                    "before trusting the denominator.")
            out.append((ns, name))
            UPSTREAM_DECL[(ns, name)] = (
                ".work/mlx/mlx/"
                + path.relative_to(ROOT / ".work/mlx/mlx").as_posix()
                + f":{text[:match.start()].count(chr(10)) + 1}")
    return out


HELPER_FAMILY = {
    "dispatch_elementwise": ["Elementwise"],
    "dispatch_int_elementwise": ["Elementwise"],
    "dispatch_comparison": ["Compare"],
    "dispatch_logical_or": ["LogicalOr"],
    "dispatch_matmul": ["Matmul"],
    "dispatch_softmax": ["Softmax"],
}

COPY_EVIDENCE = re.compile(
    r"\b(copy_gpu|copy_gpu_inplace|contiguous_copy_gpu|reshape_in_eval|"
    r"swapaxes_in_eval)\b")

DELEGATE_EVIDENCE = re.compile(r"\bomarchy::(eval_\w+)\(")
DELEGATE_FILES = {"eval_compiled_tape": "compiled.cpp"}
# The tape interpreter runs every class upstream fuses, so the op set is
# upstream's own is_unary/is_binary/is_ternary/is_broadcast predicates.
UPSTREAM_FUSABLE = ROOT / ".work/mlx/mlx/compile.cpp"
FUSABLE_REGION = re.compile(
    r"bool is_unary\(.*?bool is_noop\(", re.DOTALL)


def parse_delegate_info(entries):
    """Mine fragments and tape op names from delegate target files."""
    info = {}
    for entry in entries:
        body = strip_comments(entry.get("impl_body", ""))
        match = DELEGATE_EVIDENCE.search(body)
        if not match:
            continue
        target = DELEGATE_FILES.get(match.group(1))
        fragments, tape_ops = [], []
        if target:
            path = ROOT / "overlay/mlx/backend/omarchy" / target
            text = strip_comments(path.read_text())
            for throw_match in re.finditer(
                    r"throw std::(?:runtime_error|invalid_argument)\s*\(",
                    text):
                args = balanced(text, throw_match.end() - 1)
                rendered = ""
                for literal, name_call in re.findall(
                        r'"((?:[^"\\]|\\.)*)"|(\w+\.name\(\))', args):
                    rendered += "{name}" if name_call else literal
                if rendered not in fragments:
                    fragments.append(rendered.strip())
            region = FUSABLE_REGION.search(
                strip_comments(UPSTREAM_FUSABLE.read_text()))
            if not region:
                raise SystemExit(
                    f"{UPSTREAM_FUSABLE}: is_unary..is_noop region not "
                    "found; the fusable op set cannot be derived.")
            tape_ops = list(dict.fromkeys(re.findall(
                r"typeid\(\w+\) == typeid\((\w+)\)", region.group(0))))
        info[(entry["ns"], entry["name"])] = {
            "fragments": fragments, "tape_ops": tape_ops}
    return info


def build_row(entry, families, variant_dtypes, fragments, called, dtypes,
              bool_out_helpers, kernel_helpers, cases, delegate=None):
    name = entry["name"]
    display = f"{entry['ns']}::{name}" if entry["ns"] else name
    kinds = set(entry["kinds"])
    impl_body = entry.get("impl_body", "")

    if not impl_body and (
            "USE_FALLBACK" in kinds or entry.get("fallback_true")):
        status = "composed (fallback)"
        constraint_frags = [
            "no omarchy kernel; mlx composes core ops on the device"
        ]
        kernel_names = []
        row_dtypes = set()
    elif not impl_body and ("BINARY" in kinds or "UNARY" in kinds):
        status = "native"
        kernel_names = ["Elementwise"]
        row_dtypes = set(helper_dtypes(
            "dispatch_elementwise", dtypes, called))
        constraint_frags = helper_fragments(
            "dispatch_elementwise", fragments, called)
    elif not impl_body:
        status = "named-error"
        constraint_frags = [f"unsupported({display})"]
        kernel_names = []
        row_dtypes = set()
    else:
        body = strip_comments(impl_body)
        # An omarchy::unsupported call at statement scope (not nested
        # in any conditional block) is [[noreturn]]: every execution
        # path hits it, so any kernels below are dead code and the
        # primitive refuses all inputs however the body reads.
        depth = 0
        refuses_all = False
        for index in range(len(body)):
            char = body[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif (depth == 0
                    and (UNSUPPORTED_CALL.match(body, index)
                         or THROW_CALL.match(body, index))):
                refuses_all = True
                break
        tokens = re.findall(r"omarchy::ComputeKernel::(\w+)", body)
        helpers_called = re.findall(
            r"\b(dispatch_\w+|fill_broadcast_transport)\(", body)
        kernel_names = []
        for token in tokens:
            for base in families:
                if token == base or token.startswith(base):
                    if base not in kernel_names:
                        kernel_names.append(base)
                    break
        for helper in helpers_called:
            for base in HELPER_FAMILY.get(helper, []):
                if base not in kernel_names:
                    kernel_names.append(base)
            if helper == "dispatch_sort":
                args = re.search(r"dispatch_sort\(([^)]*)\)", body)
                base = "ArgSort" if args and re.search(
                    r"\btrue\b", args.group(1)) else "Sort"
                if base not in kernel_names:
                    kernel_names.append(base)

        row_dtypes = set()
        for token in tokens:
            row_dtypes |= variant_dtypes.get(token, set())
        for helper in helpers_called:
            row_dtypes |= helper_dtypes(helper, dtypes, called)
            if helper in bool_out_helpers:
                row_dtypes.add("bool")
        # A dtype guard in the body narrows the family coverage, as in
        # GreaterEqual accepting int32 only.
        guard = {
            DTYPE_LABELS[token]
            for token in re.findall(
                r"dtype\(\)\s*!=\s*"
                r"(float32|float16|bfloat16|int32|uint32|int64)\b", body)
        }
        if guard and row_dtypes & guard:
            row_dtypes = (row_dtypes & guard) | (row_dtypes & {"bool"})

        delegate_dtypes = set()
        delegate_kernels = []
        if delegate:
            delegate_dtypes = set(delegate.get("dtypes", ()))
            for base in delegate.get("kernels", ()):
                if base not in delegate_kernels:
                    delegate_kernels.append(base)
        row_dtypes |= delegate_dtypes
        for base in delegate_kernels:
            if base not in kernel_names:
                kernel_names.append(base)
        copy_evidence = bool(COPY_EVIDENCE.search(body))
        dispatches = (
            bool(tokens)
            or bool(kernel_names)
            or copy_evidence
            or delegate is not None
            or any(
                helper in kernel_helpers
                for helper in helpers_called))
        # Gates the entry's own eval body places above the dispatch
        # (modes, layouts, ranks) make the implementation genuinely
        # partial. Refusals that live only in shared engine helpers
        # (contiguity transport, uint32 spans, capability gates) stay
        # row constraints, not partial markers.
        own_gates = bool(extract_fragments(body))
        delegate_gates = bool(delegate and delegate.get("fragments"))
        if refuses_all:
            status = "named-error"
            kernel_names = []
            row_dtypes = set()
        elif dispatches:
            if own_gates or delegate_gates:
                status = "partial"
            elif delegate or len(set(kernel_names)) > 1 or copy_evidence:
                status = "composed"
            else:
                status = "native"
        elif own_gates:
            status = "named-error"
        else:
            status = "native"
        constraint_frags = (
            list(delegate["fragments"]) if delegate
            else extract_fragments(body))
        seen = set(constraint_frags)
        for helper in helpers_called:
            for fragment in helper_fragments(helper, fragments, called):
                if fragment not in seen:
                    seen.add(fragment)
                    constraint_frags.append(fragment)

    ordered = [
        dtype + ("*" if dtype in ("f16", "bf16") else "")
        for dtype in DTYPE_ORDER
        if dtype in row_dtypes
    ]
    anchor_link, anchor_kind = anchor_for(display, cases)
    return {
        "display": display,
        "status": status,
        "kernels": kernel_names,
        "dtypes": ordered,
        "constraints": constraint_frags,
        "anchor": anchor_link,
        "anchor_kind": anchor_kind,
        "line": entry.get("impl_line", entry["line"]),
    }


def format_families(families, copy_families, gates):
    lines = [
        "## Kernel families",
        "",
        "Variants come from `ComputeKernel` in "
        "`overlay/mlx/backend/omarchy/compute.h`. "
        "One variant exists per dtype the family serves.",
        "",
        "| Family | Variants | Dtypes | Dispatched from |",
        "|---|---|---|---|",
    ]
    for base, entry in families.items():
        dtypes = ", ".join(
            dtype + ("*" if dtype in ("f16", "bf16") else "")
            for dtype in DTYPE_ORDER if dtype in entry["dtypes"])
        source = "copy.cpp" if base in copy_families else "primitives.cpp"
        lines.append(
            f"| {base} | {len(entry['variants'])} | {dtypes} | {source} |")
    lines.append("")
    if gates:
        lines.append("Capability gates behind the `*` markers:")
        lines.append("")
        for dtype, first, second in gates:
            label = "float16" if dtype == "float16" else "bfloat16"
            lines.append(
                f"- {label} needs `{first}` and `{second}`.")
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.date.today().isoformat(),
        help="generation date recorded in the header (YYYY-MM-DD)")
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="also write the coverage badge in shields.io endpoint "
             "format (docs/coverage.json)")
    args = parser.parse_args()

    primitives_text, entries = parse_primitives()
    families, variant_dtypes = parse_kernel_families()
    gates = parse_capability_gates(primitives_text)
    helpers, fragments, called, helper_dtype_map = parse_helpers(
        primitives_text, variant_dtypes)
    kernel_helpers = kernel_reaching_helpers(helpers, called)
    upstream = parse_upstream_primitives()
    op_map, vjp_rules, trace_ops = parse_op_constructions(set(upstream))
    upstream_names = sorted({name for _, name in upstream})
    cases = parse_test_cases(
        op_map, set(upstream), upstream_names, vjp_rules, trace_ops)
    bool_out_helpers = {
        helper
        for helper, body in helpers.items()
        if re.search(r"out\.dtype\(\)\s*!=\s*bool_", body)
    }
    copy_text = strip_comments(COPY_CPP.read_text())
    copy_families = set()
    for token in re.findall(r"omarchy::ComputeKernel::(\w+)", copy_text):
        for base in families:
            if token == base or token.startswith(base):
                copy_families.add(base)
                break

    delegate_info = parse_delegate_info(entries)

    def build(entry, delegate=None):
        return build_row(entry, families, variant_dtypes, fragments,
                         called, helper_dtype_map, bool_out_helpers,
                         kernel_helpers, cases, delegate)

    core_rows = {}
    rows_by_key = {}
    for entry in entries:
        key = (entry["ns"], entry["name"])
        if key in delegate_info:
            continue
        rows_by_key[key] = build(entry)
    for key, info in delegate_info.items():
        tape_dtypes, tape_kernels = set(), []
        for op_name in info["tape_ops"]:
            for (ns, name), row in rows_by_key.items():
                if ns == "" and name == op_name:
                    tape_dtypes |= {
                        dtype.rstrip("*") for dtype in row["dtypes"]}
                    for base in row["kernels"]:
                        if base not in tape_kernels:
                            tape_kernels.append(base)
                    break
        info["dtypes"] = tape_dtypes
        info["kernels"] = tape_kernels
        entry = next(entry for entry in entries
                     if (entry["ns"], entry["name"]) == key)
        rows_by_key[key] = build(entry, delegate=info)
    # A distributed named-error row becomes a real computing path when
    # upstream's CPU backend implements the eval AND a two-rank harness
    # case value-anchors it. The omarchy backend has and needs no eval_gpu
    # for these: ring pins the communication stream to the CPU device
    # (.work/mlx/mlx/distributed/ring/ring.cpp
    # RingGroup::communication_stream), so the transport path through
    # upstream backend/cpu IS the implementation, on a Mac and here alike.
    # No multi-rank value anchor, no flip: the named-error row stays.
    transport_cpu_real, _ = parse_backend_eval_impls(CPU_BACKEND, "cpu")
    for key, row in rows_by_key.items():
        if (key[0] == DISTRIBUTED_NS
                and row["status"] == "named-error"
                and key[1] in transport_cpu_real
                and row["anchor_kind"] == "value"):
            row["status"] = "composed (cpu transport)"
            row["constraints"] = [
                "runs on the ring CPU communication stream through "
                "upstream backend/cpu; never dispatches an omarchy kernel"]
    rows = [rows_by_key[(entry["ns"], entry["name"])]
            for entry in entries]

    backend_keys = {(entry["ns"], entry["name"]) for entry in entries}
    phantom = backend_keys - set(upstream)
    if phantom:
        raise SystemExit(
            "backend entries absent from upstream MLX: "
            + ", ".join(
                f"{ns}::{name}" if ns else name
                for ns, name in sorted(phantom)))
    no_entry = sorted(set(upstream) - backend_keys)
    shared_gpu_names = parse_shared_gpu_primitives()
    shared_gpu = [key for key in no_entry if key[1] in shared_gpu_names]
    nothing_runs = [key for key in no_entry if key[1] not in shared_gpu_names]
    total = len(upstream)

    def share(count):
        return f"{count / total * 100:.1f}%"

    shared_links = {
        key: anchor_for(
            f"{key[0]}::{key[1]}" if key[0] else key[1], cases)
        for key in shared_gpu
    }
    shared_counts = {"value": 0, "pin": 0, None: 0}
    for _, kind in shared_links.values():
        shared_counts[kind] += 1

    # Per-upstream-key coverage, so the Mac denominator can reuse the
    # same evidence the bucket loop counts.
    def covered_upstream_key(key):
        if key in rows_by_key:
            row = rows_by_key[key]
            return row["anchor_kind"] == "value" and (
                row["status"] in ("native", "partial")
                or row["status"].startswith("composed"))
        if key in shared_gpu:
            return shared_links[key][1] == "value"
        return False

    covered_keys = {key for key in upstream if covered_upstream_key(key)}

    status_families = (
        ("native", lambda row: row["status"] == "native"),
        ("composed", lambda row: row["status"].startswith("composed")),
        ("partial", lambda row: row["status"] == "partial"),
    )
    anchor_kinds = (
        ("value-tested", "value"),
        ("error-pins only", "pin"),
        ("untested", None),
    )
    buckets = []
    covered = 0
    computing_uncovered = 0
    for family_name, pred in status_families:
        for label, kind in anchor_kinds:
            count = sum(
                1 for row in rows
                if pred(row) and row["anchor_kind"] == kind)
            buckets.append((f"{family_name}, {label}", count))
            if kind == "value":
                covered += count
            else:
                computing_uncovered += count
    for label, kind in (
            ("shared-gpu, value-tested", "value"),
            ("shared-gpu, error-pins only", "pin"),
            ("shared-gpu, untested", None)):
        buckets.append((label, shared_counts[kind]))
        if kind == "value":
            covered += shared_counts[kind]
        else:
            computing_uncovered += shared_counts[kind]
    named_error = sum(
        1 for row in rows if row["status"] == "named-error")
    named_error_pinned = sum(
        1 for row in rows
        if row["status"] == "named-error"
        and row["anchor_kind"] is not None)
    buckets.append(
        ("named-error (refuses all inputs)", named_error))
    buckets.append(
        ("not implemented (nothing runs)", len(nothing_runs)))
    implemented = covered + computing_uncovered
    coverage_pct = covered / total * 100
    if len(covered_keys) != covered:
        raise SystemExit(
            f"internal: bucket coverage {covered} != per-key coverage "
            f"{len(covered_keys)}; the Mac denominator would drift from "
            "the reported buckets.")
    metal_gpu_real, metal_gpu_throws = parse_backend_eval_impls(
        METAL_BACKEND, "gpu")
    mac_impl_cite = {**metal_gpu_real}
    for directory in (CPU_BACKEND, COMMON_BACKEND):
        cpu_real, _ = parse_backend_eval_impls(directory, "cpu")
        mac_impl_cite.update(cpu_real)
    # Mac-usable rule: a primitive is in the Mac denominator when
    # upstream's Metal backend gives it a real eval_gpu, or when
    # upstream's CPU backend (which ships on macOS) gives it a real
    # eval_cpu. Otherwise a Mac user cannot execute it on any stream.
    # An op-layer short-circuit excludes nothing by itself: it only
    # changes how the user reaches the primitive, unless no
    # Mac-reachable input constructs it at all.
    mac_excluded = {}
    for key in upstream:
        if key[1] in mac_impl_cite:
            continue
        mac_excluded[key] = (
            metal_gpu_throws.get(key[1]) or UPSTREAM_DECL[key])
    mac_total = total - len(mac_excluded)
    mac_covered = len(covered_keys - set(mac_excluded))
    mac_pct = mac_covered / mac_total * 100
    mac_unclosable = sorted(
        name for name in UNCLOSABLE_ON_VULKAN if name in mac_impl_cite)
    mac_ceiling_count = mac_total - len(mac_unclosable)

    def mac_marker(display):
        ns, _, name = display.rpartition("::")
        key = (ns, name)
        if key in mac_excluded:
            return f"no (`{mac_excluded[key]}`)"
        return "yes"


    # Upstream's own refusals, display-only. The Metal backend throws
    # for these primitives too, so the gap is not omarchy-specific;
    # the annotation never changes the covered count or denominator.
    metal_refusals = parse_upstream_metal_refusals(
        {name for _, name in upstream})
    upstream_refused = [
        (row["display"], row["status"])
        for row in rows
        if row["display"].rpartition("::")[2] in metal_refusals
        and row["status"] == "named-error"]
    upstream_refused.extend(
        (f"{ns}::{name}" if ns else name,
         "not implemented (nothing runs)")
        for ns, name in nothing_runs
        if name in metal_refusals)
    singleton_short_circuit = parse_distributed_singleton_guard()

    def badge_color(pct):
        if pct < 34:
            return "red"
        if pct < 67:
            return "orange"
        if pct < 90:
            return "yellow"
        return "green"

    if args.json_out:
        payload = {
            "schemaVersion": 1,
            "label": "Primitive coverage",
            "message":
                f"{mac_pct:.1f}% ({mac_covered}/{mac_total} "
                "Mac-usable primitives)",
            "color": badge_color(mac_pct),
        }
        pathlib.Path(args.json_out).write_text(
            json.dumps(payload) + "\n")

    out = []
    out.append("# Compatibility matrix")
    out.append("")
    sha = git("rev-parse", "--short", "HEAD")
    branch = git("branch", "--show-current")
    # The header records the canonical invocation so the document is
    # byte-identical no matter where --json-out pointed.
    command = (
        "python3 tools/gen-compat-matrix.py"
        " --json-out docs/coverage.json > docs/compatibility-matrix.md")
    out.append(
        f"Generated by `{command}` on {args.date} "
        f"from the working tree at commit `{sha}` on branch `{branch}`.")
    out.append(
        "Regenerate after backend changes; the generator is the source "
        "of truth and the same tree produces the same bytes.")
    out.append("")
    out.append("## Coverage")
    out.append(
        f"**Value-tested primitive coverage: {mac_pct:.1f}% "
        f"({mac_covered}/{mac_total} Mac-usable primitives). Not functional parity.**")
    out.append("")
    out.append(
        f"Against the full upstream primitive list: {coverage_pct:.1f}% "
        f"— {covered} of {total} upstream primitives. Both numbers "
        "come from the same coverage evidence below; only the "
        "denominators differ.")
    out.append("")
    out.append(
        "Coverage counts a primitive only when the omarchy backend has "
        "a real computing path for it — native kernels, a composition, "
        "or the fallback path; an eval body whose only reachable "
        "outcome is `omarchy::unsupported` is a refusal, not a path — "
        "and a TEST_CASE in `overlay/tests/omarchy/` verifies values "
        "for it: the case reaches the primitive through an op call or "
        "a direct construction and asserts results against "
        "host-computed references. Cases that only pin refusal "
        "messages, only assert an eval raised nothing, or carry a "
        "loud SKIP marker anchor nothing and are bucketed separately "
        "below. One namespace-specific bar: a `distributed::` "
        "primitive anchors only from a case whose own body asserts a "
        "multi-rank group (`group.size() == 2`), because upstream "
        "short-circuits every distributed op at `group.size() == 1` "
        "and a value assertion in a single-process run verifies that "
        "short-circuit, never the primitive.")
    out.append("")
    out.append(
        "Partial primitives count: a value-tested path does not establish "
        "support for every dtype, layout, or mode upstream accepts. "
        "The rows below list known refusals. This source-and-test-anchor "
        "inventory is not a current full-suite pass rate or a measure "
        "of complete functional compatibility.")
    out.append("")
    out.append(
        "The full-upstream denominator is every concrete primitive "
        "class upstream MLX defines — parsed from "
        "`.work/mlx/mlx/primitives.h`, `.work/mlx/mlx/fast_primitives.h`,"
        " and `.work/mlx/mlx/distributed/primitives.h` — not only the "
        f"{len(rows)} entries the omarchy backend enumerates.")
    out.append("")
    out.append(
        "The Mac denominator is derived mechanically from upstream's "
        "own sources, never from a hand-maintained list. A primitive "
        "is Mac-usable when upstream's Metal backend gives it a real "
        "`eval_gpu` body, or when upstream's CPU backend (which ships "
        "on macOS) gives it a real `eval_cpu` body. An `eval_gpu` "
        "whose body is one unconditional throw is a refusal, not an "
        "implementation. A body that checks arguments first and then "
        "computes is real. Every exclusion below cites the exact file "
        "and line; no citation, no exclusion.")
    out.append("")
    out.append(
        "Op-layer short-circuits exclude nothing by themselves. The "
        "rule: a short-circuit only changes how a user reaches the "
        "primitive, unless no Mac-reachable input constructs it at "
        "all. The distributed ops return the input unchanged at "
        "`group.size() == 1` (`.work/mlx/mlx/distributed/ops.cpp`), "
        "but a Mac user can form larger groups and "
        "`.work/mlx/mlx/backend/cpu/distributed.cpp` implements all "
        "five `eval_cpu` paths, so they stay. Conjugate, Real, and "
        "Imag are constructed only for complex input "
        "(`.work/mlx/mlx/ops.cpp:6529`, `:6744`, `:6751`), and "
        "complex dtypes are Mac-reachable, so they stay too.")
    out.append("")
    out.append(
        "Distributed primitives count only through the two-rank "
        "harness. The harness lives in "
        "`overlay/tests/omarchy/distributed/`: `test_two_rank.cpp` is "
        "one rank's doctest process, and `run-two-rank.sh` launches it "
        "twice with `MLX_RANK=0` and `MLX_RANK=1` against a generated "
        "localhost hostfile and requires both processes green. Every "
        "case in the harness asserts the two-rank group inside its own "
        "body, so a lone run of the binary fails its group-size guards "
        "instead of passing vacuously — that in-body assertion is the "
        "mechanical line between a two-rank anchor and a single-rank "
        "case that proves nothing. Their rows read `composed (cpu "
        "transport)`: ring pins the communication stream to the CPU "
        "device, so these primitives evaluate through upstream's "
        "`backend/cpu` on a Mac and here alike and never dispatch an "
        "omarchy kernel — the transport is the implementation, and the "
        "harness is the proof.")
    if mac_excluded:
        out.append("")
        out.append(
            f"### Not Mac-usable — {len(mac_excluded)} primitives")
        out.append("")
        out.append(
            "Metal refuses these and upstream's CPU backend has no "
            "`eval_cpu` for them, so no Mac user can execute them on "
            "any stream. They are outside the Mac denominator and "
            "stay in the full-upstream denominator. Note what this "
            "does to the score: this backend implements and "
            "value-tests all of them, and Metal does not, so the "
            "redefinition removes covered primitives from the "
            "numerator and denominator alike. It lowers this "
            "backend's number. The denominator was not curated to "
            "flatter anyone.")
        out.append("")
        out.append("| Primitive | Why | Citation |")
        out.append("|---|---|---|")
        for key in sorted(mac_excluded):
            ns, name = key
            display = f"{ns}::{name}" if ns else name
            out.append(
                f"| {display} | Metal `eval_gpu` throws and no CPU "
                f"`eval_cpu` exists | `{mac_excluded[key]}` |")
    out.append("")
    if mac_unclosable:
        out.append(
            "The achievable ceiling: "
            + ", ".join(f"`fast::{name}`" for name in mac_unclosable)
            + " is not implementable on Vulkan. Perfect achievable Mac "
            f"parity is therefore {mac_ceiling_count}/{mac_total} = "
            f"{mac_ceiling_count / mac_total * 100:.1f}%.")
        out.append("")
    out.append("| Bucket | Count | Share of upstream | Share of Mac-usable |")
    out.append("|---|---|---|---|")
    for label, count in buckets:
        out.append(
            f"| {label} | {count} | {share(count)} | "
            f"{count / mac_total * 100:.1f}% |")
    out.append(
        f"| upstream total | {total} | 100.0% | "
        f"{total / mac_total * 100:.1f}% |")
    out.append(
        f"| Mac-usable total | {mac_total} | "
        f"{mac_total / total * 100:.1f}% | 100.0% |")
    out.append("")
    out.append(
        "Implementation-path coverage without the value-test "
        f"requirement: {implemented}/{total} = "
        f"{implemented / total * 100:.1f}%. "
        f"{named_error_pinned} named-error primitives pin their "
        "unsupported error with a test; a pinned error is not "
        "coverage.")
    if upstream_refused:
        out.append("")
        out.append(
            f"Of the {total - covered} primitives outside coverage, "
            f"{len(upstream_refused)} are also unimplemented in "
            "upstream's own Metal backend (listed at the end of this "
            "document).")
    out.append("")
    out.append("## Status terms")
    out.append("")
    out.append(
        "- native: one omarchy kernel or engine path dispatches the "
        "operation.")
    out.append(
        "- composed: the operation assembles on the device from "
        "several dispatched kernels, or mlx composes core ops because "
        "use_fallback returns true.")
    out.append(
        "- composed (cpu transport): the primitive computes on the "
        "ring group's CPU communication stream through upstream's own "
        "`backend/cpu/distributed.cpp`, with GPU tensors shuttled by "
        "the scheduler's cross-stream copies — the same path a Mac "
        "user runs. It never dispatches an omarchy kernel, and it "
        "counts as covered only through the two-rank harness.")
    out.append(
        "- partial: a real kernel path computes for the dtypes and "
        "shapes in the row, while other caller-reachable inputs — "
        "modes, layouts, ranks, row lengths — hit the named refusals "
        "listed in the row. A partial primitive counts as covered "
        "only through a value-tested case.")
    out.append(
        "- named-error: every input refuses; eval_gpu raises "
        "`omarchy::unsupported` and the run stops with that "
        "qualifier. This includes eval bodies that keep kernels below "
        "an unconditional refusal statement — those kernels are dead "
        "code.")
    out.append(
        "- shared-gpu: the primitive has no omarchy entry, and "
        "upstream's shared `backend/gpu/primitives.cpp` resolves it "
        "generically — zero-copy buffer views or the omarchy copy "
        "engine. It counts as covered only when a TEST_CASE "
        "value-anchors it.")
    out.append(
        "- not implemented (nothing runs): upstream MLX defines the "
        "primitive, the omarchy backend has no entry for it, and the "
        "shared GPU layer has no eval_gpu for it either.")
    out.append(
        "- Test anchor kinds: a value-tested case compares results "
        "against host-computed references; a case marked `pins "
        "refusal, no values` asserts refusal messages or no-error "
        "smoke only; a loud SKIP marker in a case withdraws its value "
        "credit.")
    out.append(
        "- Dtypes marked `*` pass the capability gates listed under "
        "the kernel families. Refusal fragments show the resolved "
        "operation name.")
    out.append("")
    out.extend(format_families(families, copy_families, gates))
    out.append("## Primitives")
    out.append("")
    out.append(
        "One row per primitive entry in the Omarchy backend sources, "
        "in source order.")
    out.append("")
    out.append(
        "| Primitive | Status | Kernels | Dtypes | Named-error "
        "constraints | Test anchor | Mac |")
    out.append("|---|---|---|---|---|---|---|")
    for row in rows:
        kernels = (
            ", ".join(row["kernels"]) if row["kernels"] else "—")
        dtypes = ", ".join(row["dtypes"]) if row["dtypes"] else "—"
        row_name = row["display"].rpartition("::")[2]
        constraints = (
            "; ".join(
                f"`{fragment.replace('{name}', row_name)}`"
                for fragment in row["constraints"])
            if row["constraints"] else "—")
        out.append(
            f"| {row['display']} | {row['status']} | {kernels} | "
            f"{dtypes} | {constraints} | {row['anchor']} | "
            f"{mac_marker(row['display'])} |")
    out.append("")
    out.append(
        "Every dispatch path also bounds element counts, item offsets, "
        "and index spans to `uint32` and raises a named error beyond. "
        "No primitive falls back to CPU tensors.")
    out.append("")
    out.append(
        f"## No omarchy entry — {len(no_entry)} primitives")
    out.append("")
    out.append(
        "Upstream MLX defines these primitives, but "
        "the Omarchy backend sources have no entry for "
        "them. They count in both denominators unless marked not "
        "Mac-usable.")
    out.append("")
    out.append("| Primitive | Status | Test anchor | Mac |")
    out.append("|---|---|---|---|")
    for ns, name in no_entry:
        display = f"{ns}::{name}" if ns else name
        if (ns, name) in shared_gpu:
            status = "shared-gpu"
            anchor = shared_links[(ns, name)][0]
        else:
            status = "not implemented (nothing runs)"
            anchor = "—"
        out.append(
            f"| {display} | {status} | {anchor} | "
            f"{mac_marker(display)} |")
    out.append("")
    if upstream_refused:
        out.append(
            f"## Also unimplemented upstream — "
            f"{len(upstream_refused)} primitives")
        out.append("")
        out.append(
            "Upstream's Metal backend throws for these primitives in "
            "`.work/mlx/mlx/backend/metal/distributed.cpp`"
            + ("; the distributed op layer also short-circuits each "
               "at group.size() == 1 in "
               "`.work/mlx/mlx/distributed/ops.cpp`, so the primitive "
               "is never constructed in a single-rank run"
               if singleton_short_circuit else "")
            + ". The omarchy named errors match upstream's own "
              "refusal. These rows stay in the denominator and count "
              "zero toward coverage. Upstream's CPU backend implements "
              "each `eval_cpu`, so a Mac user still reaches them and "
              "they stay in the Mac denominator.")
        out.append("")
        out.append("| Primitive | Omarchy status |")
        out.append("|---|---|")
        for display, status in upstream_refused:
            out.append(f"| {display} | {status} |")
        out.append("")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
