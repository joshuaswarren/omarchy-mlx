#!/usr/bin/env python3
"""Query the mlx-omarchy community hardware dataset.

Stdlib-only CLI for coding agents and humans. It reads the mirrored
snapshot on the `community-data` branch, or the live public endpoint, and
answers: which machines reported, what ran there, and how benchmark
numbers compare across devices.

Usage:
  query_community_data.py [filters] list
  query_community_data.py show SHA256_PREFIX
  query_community_data.py [filters] compare [--metric tflops] [--size N]

Fleet-gap filters: --cpu-present N, --hotplug yes|no, --ane yes|no,
--shortfall yes|no, --boot SUBSTR, --cmdline SUBSTR. list lines show
cores=online/present, ane=dt|- markers, and core-shortfall when set.
Sources (--source):
  auto    local snapshot when present, else the live endpoint (default)
  local   mirrored snapshot only
  remote  live public endpoint only

Local snapshot ladder, first hit wins:
  --snapshot DIR           directory holding latest.jsonl
  $MLX_OMARCHY_DATA_DIR    same, from the environment
  <repo>/community-data/snapshot/latest.jsonl
  origin/community-data:snapshot/latest.jsonl via `git show`

Exit codes: 0 ok, 1 no dataset or endpoint unreachable. Partial records
and unknown schema_version values never crash the tool; malformed JSONL
lines are skipped and counted.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://mlx-omarchy-community-data.joshua-s-warren.workers.dev"
# Cloudflare bot protection 403s default Python user agents on workers.dev.
USER_AGENT = ("mlx-omarchy-community-data/1.0 "
              "(+https://github.com/joshuaswarren/omarchy-mlx)")
SNAPSHOT_RELPATH = Path("community-data") / "snapshot"

METRIC_UNITS = {"tflops": "TFLOP/s", "median_ms": "ms", "min_ms": "ms",
                "max_ms": "ms"}


class DatasetError(RuntimeError):
    """User-facing failure: no dataset, bad source, endpoint down."""


# ---------------------------------------------------------------------------
# record accessors - tolerant across partial payloads and schema drift
# ---------------------------------------------------------------------------

def _payload(record):
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _walk(mapping, *path):
    """Final value at path, whatever its type; None on any miss."""
    current = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _dig(mapping, *path):
    """Scalar at path; None when missing or a container."""
    value = _walk(mapping, *path)
    if value is None or isinstance(value, (dict, list)):
        return None
    return value


def _dig_dict(mapping, *path):
    """Dict at path; {} when missing or not a dict."""
    value = _walk(mapping, *path)
    return value if isinstance(value, dict) else {}


def _first(record, paths):
    """First scalar hit among paths, checking the record then payload."""
    payload = _payload(record)
    for path in paths:
        value = _dig(record, *path)
        if value is None:
            value = _dig(payload, *path)
        if value not in (None, ""):
            return value
    return None


def record_sha(record):
    for key in ("content_sha256", "sha256"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def record_kind(record):
    for value in (record.get("kind"), _payload(record).get("kind"),
                  _payload(record).get("report")):
        if isinstance(value, str) and value:
            return value
    return "unknown"


def record_chip(record):
    value = _first(record, [
        ("host", "chip"), ("host", "devicetree", "model"),
        ("host", "model"), ("chip",), ("soc",), ("soc_name",),
    ])
    if value:
        return str(value)
    device = _dig_dict(_payload(record), "mesa", "gpu").get("deviceName")
    return str(device) if device else None


def record_kernel(record):
    value = _first(record, [
        ("host", "kernel_release"), ("host", "kernel"),
        ("kernel",), ("kernel_version",),
    ])
    return str(value) if value else None


def _mesa_package_version(record):
    packages = _dig_dict(_payload(record), "mesa_package")
    for channel in ("pacman", "dpkg"):
        stdout = _dig(_dig_dict(packages, channel), "stdout")
        if not isinstance(stdout, str):
            continue
        for line in stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == "mesa" and len(parts) > 1:
                return parts[1]
    return None


def record_mesa(record):
    """Display string: driver name plus the freshest version we can find.

    The read API returns flat summary fields (mesa_driver, mesa_device),
    while a local archive payload nests them under mesa.gpu. Check the
    flat fields first: they are what /v1/results serves.
    """
    gpu = _dig_dict(_payload(record), "mesa", "gpu")
    driver = _first(record, [("mesa_driver",), ("mesa", "driver")]) \
        or gpu.get("driverName")
    device = _first(record, [("mesa_device",)]) or gpu.get("deviceName")
    version = _mesa_package_version(record) or gpu.get("driverVersion")
    if driver and version:
        return f"{driver} {version}"
    for value in (driver, version, device):
        if value:
            return str(value)
    return None


def record_mesa_blob(record):
    """Lowercased searchable text for --mesa: driver, device, API, package."""
    gpu = _dig_dict(_payload(record), "mesa", "gpu")
    parts = [str(gpu.get(k)) for k in ("driverName", "deviceName",
                                       "apiVersion") if gpu.get(k)]
    for path in (("mesa_driver",), ("mesa_device",)):
        value = _first(record, [path])
        if value:
            parts.append(str(value))
    package = _mesa_package_version(record)
    if package:
        parts.append(package)
    return " ".join(parts).lower()


def record_mlx_version(record):
    value = _first(record, [("mlx_version",)]) or _first(_payload(record), [
        ("mlx", "distributions", "mlx-omarchy"), ("mlx", "version"),
        ("mlx_omarchy_version",), ("version",),
    ])
    return str(value) if value else None


def record_cpu_online(record):
    value = _first(record, [("cpu_online",), ("host", "cpu_online"),
                            ("host", "cpu", "online")])
    return value if isinstance(value, int) else None


def record_cpu_present(record):
    value = _first(record, [("cpu_present",), ("host", "cpu", "present")])
    return value if isinstance(value, int) else None


def record_hotplug(record):
    value = _first(record, [("hotplug_control",),
                            ("host", "cpu", "hotplug_control")])
    return value if isinstance(value, bool) else None


def record_core_shortfall(record):
    """True/False for the unexplained fewer-cores-than-present fact."""
    nested = _walk(_payload(record), "host", "core_shortfall")
    if isinstance(nested, dict):
        return True
    value = _first(record, [("core_shortfall",)])
    if isinstance(value, bool):
        return value
    # Older nested reports carry only the counts: derive, still
    # respecting a maxcpus/nosmp clamp in the nested command line.
    cpu = _dig_dict(_payload(record), "host", "cpu")
    present, online = cpu.get("present"), cpu.get("online")
    if isinstance(present, int) and isinstance(online, int) \
            and present > online:
        cmdline = _dig(_payload(record), "host", "cmdline") or ""
        if not re.search(r"\bmaxcpus=\d+\b|\bnosmp\b", cmdline):
            return True
    return None


def record_ane_dt(record):
    """(node_present, compatible_blob) across flat and nested shapes."""
    node = _first(record, [("ane_dt_node",), ("ane", "devicetree", "node")])
    compat = _first(record, [("ane_dt_compatible",)])
    if compat is None:
        nested = _walk(_payload(record), "ane", "devicetree", "compatible")
        if isinstance(nested, list):
            compat = " ".join(str(t) for t in nested)
    return (node if isinstance(node, bool) else None,
            str(compat) if compat else None)


def record_boot_chain(record):
    value = _first(record, [("boot_chain",)])
    if value:
        return str(value)
    boot = _dig_dict(_payload(record), "host", "boot")
    chain = " ".join(f"{key}={boot[key]}" for key in sorted(boot)
                     if isinstance(boot.get(key), str) and boot[key])
    return chain or None


def record_cmdline(record):
    value = _first(record, [("cmdline",), ("host", "cmdline")])
    return str(value) if value else None


def record_benchmarks(record):
    """[(label, metric, size, value)] from known benchmark locations.

    Two shapes: the read API serves a flat list of rows on the record,
    while a deep archive payload nests them under benchmark.matmul.
    Without the flat shape, comparing devices over the public dataset
    returns nothing, which is the whole point of the dataset.
    """
    entries = record.get("benchmark")
    if not isinstance(entries, list):
        benchmark = _dig_dict(_payload(record), "benchmark")
        entries = benchmark.get("matmul") if isinstance(benchmark, dict) \
            else None
    out = []
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        size = entry.get("n")
        for metric in ("tflops", "median_ms", "min_ms", "max_ms"):
            value = entry.get(metric)
            if isinstance(value, (int, float)):
                out.append((f"matmul@{size}", metric, size, float(value)))
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def parse_jsonl(text):
    """(records, skipped) from JSONL text; bad lines never raise."""
    records, skipped = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(obj, dict):
            records.append(obj)
        else:
            skipped += 1
    return records, skipped


def snapshot_file(snapshot_dir):
    return Path(snapshot_dir) / "latest.jsonl"


def load_local(args, repo_root):
    """Return (records, skipped, source_label) from the local ladder.

    An explicit --snapshot is exact: if that directory holds no dataset
    the call fails there. Falling through to the mirrored branch would
    silently answer from a DIFFERENT dataset than the one named, which
    is worse than an error.
    """
    tried = []
    if args.snapshot:
        path = snapshot_file(Path(args.snapshot))
        tried.append(str(path))
        if path.is_file():
            records, skipped = parse_jsonl(
                path.read_text(encoding="utf-8", errors="replace"))
            return records, skipped, str(path)
        raise DatasetError(
            "no dataset at the requested --snapshot. Looked at: " +
            ", ".join(tried) + ". Point --snapshot at a directory holding "
            "latest.jsonl, drop it to search the repository and the "
            "mirrored community-data branch, or use --source remote.")

    candidates = []
    env_dir = os.environ.get("MLX_OMARCHY_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(repo_root / SNAPSHOT_RELPATH)
    for directory in candidates:
        path = snapshot_file(directory)
        tried.append(str(path))
        if path.is_file():
            records, skipped = parse_jsonl(
                path.read_text(encoding="utf-8", errors="replace"))
            return records, skipped, str(path)
    for ref in ("origin/community-data", "community-data"):
        spec = f"{ref}:snapshot/latest.jsonl"
        tried.append(f"git {spec}")
        try:
            proc = subprocess.run(["git", "show", spec], cwd=repo_root,
                                  capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            break
        if proc.returncode == 0:
            records, skipped = parse_jsonl(proc.stdout.decode("utf-8",
                                                             "replace"))
            return records, skipped, f"git {spec}"
    raise DatasetError(
        "no local dataset found. Looked at: " + ", ".join(tried) +
        ". Fetch the community-data branch (git fetch origin "
        "community-data), pass --snapshot DIR, or use --source remote.")


def fetch_url(url, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise DatasetError(f"HTTP {exc.code} from {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DatasetError(f"endpoint unreachable: {exc}") from exc


def load_remote(args):
    """(records, skipped, source_label) from the live JSONL endpoint."""
    base = args.base_url.rstrip("/")
    text = fetch_url(f"{base}/v1/dataset/latest.jsonl").decode(
        "utf-8", "replace")
    records, skipped = parse_jsonl(text)
    return records, skipped, f"{base}/v1/dataset/latest.jsonl"


def load_dataset(args, repo_root):
    if args.source == "local":
        return load_local(args, repo_root)
    if args.source == "remote":
        return load_remote(args)
    try:
        return load_local(args, repo_root)
    except DatasetError:
        return load_remote(args)


# ---------------------------------------------------------------------------
# read-side correction: macOS phandle byte-swap (pre-v0.6.5 rows)
# ---------------------------------------------------------------------------

# The v0.6.4 collector decoded the macOS ioreg `AAPL,phandle` OSData
# big-endian; ioreg presents cells in HOST byte order (little-endian on
# arm64), so these rows store every phandle 32-bit byte-reversed (true
# 0x169 stored as 0x69010000). Fixed in v0.6.5 (commit 7b05e936). The
# stored rows are immutable (no delete route) and carry no
# collector-version field that could identify them (`mlx_version` is the
# MLX build, `source_commit` is null), so the affected rows are pinned
# here by their content sha256: an enumerated, auditable allow-list,
# NOT a value-shape heuristic (a magnitude test would silently corrupt
# a legitimately large phandle). Identified by the 2026-09-17 audit of
# every macOS row in the live dataset, node by node.
_PHANDLE_BYTESWAP_ROWS = frozenset({
    "14fbcbe675891e66dbd852a51c3fef594a0b71232568403d697cdf7f51d98850",  # M1 Ultra t6002
    "5e75a04c972708447fa80c7ba989d4c2861103328b6bab43cb10a37a63a6aa71",  # M1 Ultra t6002
    "857e1e6c61b07e57cc05bc5c46e0cb6535a6d3dd460e0eae256cb7e581fa7dd8",  # M2 Pro t6020
    "869e1c8ba779b973a60cf0f801174070381d370e8cc71196aeff0e8543ed167c",  # M1 Ultra t6002
    "a3e974f8e6f98febee4635fc8c381114d8ad0ecd979b0a29371e7e4bc3b07a0c",  # M2 Max t6021
})


def _correct_swapped_phandle(record):
    """Deterministic read-side repair for the 5 known-affected rows.

    Byte-reverses `macos.ane_nodes[*].phandle` and
    `macos.dart_nodes[*].phandle` in place and returns a consumer-visible
    note; None (no touch) for every other row. 0x69010000 -> 0x169.
    """
    sha = record_sha(record)
    if sha not in _PHANDLE_BYTESWAP_ROWS:
        return None
    detail = _walk(record, "summary", "ane_port_detail") \
        or _walk(record, "ane_port_detail")
    macos = detail.get("macos") if isinstance(detail, dict) else None
    if not isinstance(macos, dict):
        return None
    fixed = 0
    for group in ("ane_nodes", "dart_nodes"):
        for node in macos.get(group) or []:
            ph = node.get("phandle") if isinstance(node, dict) else None
            if not isinstance(ph, int) or ph < 0 or ph > 0xFFFFFFFF:
                continue
            node["phandle"] = int.from_bytes(ph.to_bytes(4, "big"),
                                             "little")
            fixed += 1
    if not fixed:
        return None
    return (f"read-side correction: {fixed} phandle value(s) byte-reversed "
            f"(pre-v0.6.5 macOS byte-swap, row {sha[:12]}); stored values "
            f"were 32-bit reversed")


# ---------------------------------------------------------------------------
# filtering and commands
# ---------------------------------------------------------------------------


def _tri(value):
    """--flag yes/no/absent to True/False/None."""
    return None if value is None else value == "yes"


def record_matches(record, filters):
    chip_blob = " ".join(
        x for x in (record_chip(record),
                    str(_dig(_payload(record), "mesa", "gpu", "deviceName")
                        or "")) if x).lower()
    checks = (
        ("kind", record_kind(record)),
        ("chip", chip_blob),
        ("kernel", (record_kernel(record) or "").lower()),
        ("mesa", record_mesa_blob(record)),
        ("mlx_version", (record_mlx_version(record) or "").lower()),
    )
    for attr, haystack in checks:
        needle = getattr(filters, attr)
        if needle and needle.lower() not in haystack:
            return False
    present = getattr(filters, "cpu_present", None)
    if present is not None and record_cpu_present(record) != present:
        return False
    hotplug = _tri(getattr(filters, "hotplug", None))
    if hotplug is not None and record_hotplug(record) is not hotplug:
        return False
    ane = _tri(getattr(filters, "ane", None))
    if ane is not None:
        node, _compat = record_ane_dt(record)
        if node is not ane:
            return False
    shortfall = _tri(getattr(filters, "shortfall", None))
    if shortfall is not None \
            and record_core_shortfall(record) is not shortfall:
        return False
    for attr, value in (("boot", record_boot_chain(record)),
                        ("cmdline", record_cmdline(record))):
        needle = getattr(filters, attr, None)
        if needle and needle.lower() not in (value or "").lower():
            return False
    return True


def filter_records(records, filters):
    return [r for r in records if record_matches(r, filters)]


def one_line(record):
    benches = record_benchmarks(record)
    return "  ".join(filter(None, (
        (record_sha(record) or "?")[:12],
        f"kind={record_kind(record)}",
        f"chip={record_chip(record) or 'unknown'}",
        f"kernel={record_kernel(record) or 'unknown'}",
        f"mesa={record_mesa(record) or 'unknown'}",
        f"mlx-omarchy={record_mlx_version(record) or 'unknown'}",
        f"bench={len(benches)}" if benches else "",
        f"cores={record_cpu_online(record) or '?'}/"
        f"{record_cpu_present(record) or '?'}"
        if record_cpu_present(record) is not None else "",
        "ane=dt" if record_ane_dt(record)[0] is True
        else "ane=-" if record_ane_dt(record)[0] is False else "",
        "core-shortfall" if record_core_shortfall(record) is True else "",
        f"boot={record_boot_chain(record)}" if record_boot_chain(record)
        else "",
    )))


def find_record(records, prefix):
    matches = [r for r in records
               if (record_sha(r) or "").startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DatasetError(f"no record with hash prefix {prefix!r}")
    raise DatasetError(f"hash prefix {prefix!r} is ambiguous "
                       f"({len(matches)} records)")


def cmd_list(records, skipped, source, as_json):
    if as_json:
        print(json.dumps({"source": source, "count": len(records),
                          "skipped_malformed": skipped,
                          "results": records}, indent=2, sort_keys=True))
        return 0
    if not records:
        print(f"no records match ({source}).")
        return 0
    for record in records:
        print(one_line(record))
    if skipped:
        print(f"({skipped} malformed line(s) skipped)", file=sys.stderr)
    return 0


def cmd_show(records, prefix, args):
    if args.source == "remote":
        base = args.base_url.rstrip("/")
        body = fetch_url(f"{base}/v1/results/{prefix}")
        record = json.loads(body.decode("utf-8"))
    else:
        record = find_record(records, prefix)
    note = _correct_swapped_phandle(record)
    if note:
        # Visible in both worlds: a parsed key for JSON consumers, a
        # stderr line for humans - the values below are adjusted.
        record["_read_side_correction"] = note
        print(note, file=sys.stderr)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def compare_groups(records, metric, size):
    """[{device, count, min, median, max}] sorted by device name."""
    groups = {}
    for record in records:
        device = record_chip(record) or "unknown"
        for _, bench_metric, bench_size, value in record_benchmarks(record):
            if bench_metric != metric:
                continue
            if size is not None and bench_size != size:
                continue
            groups.setdefault(device, []).append(value)
    out = []
    for device in sorted(groups):
        values = groups[device]
        out.append({"device": device, "count": len(values),
                    "min": min(values),
                    "median": statistics.median(values),
                    "max": max(values)})
    return out


def cmd_compare(records, metric, size, as_json):
    groups = compare_groups(records, metric, size)
    if as_json:
        print(json.dumps({"metric": metric,
                          "unit": METRIC_UNITS.get(metric, ""),
                          "size": size, "groups": groups},
                         indent=2, sort_keys=True))
        return 0
    if not groups:
        print(f"no {metric} benchmark data in the selected records.")
        return 0
    print(f"metric {metric} ({METRIC_UNITS.get(metric, '?')}), "
          f"{len(groups)} device(s):")
    print(f"{'device':<28}{'n':>4}{'min':>12}{'median':>12}{'max':>12}")
    for group in groups:
        print(f"{group['device'][:27]:<28}{group['count']:>4}"
              f"{group['min']:>12.6g}{group['median']:>12.6g}"
              f"{group['max']:>12.6g}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="query_community_data.py",
        description="Query the mlx-omarchy community hardware dataset.")
    parser.add_argument("--source", choices=("auto", "local", "remote"),
                        default="auto")
    parser.add_argument("--snapshot", default=None,
                        help="directory holding latest.jsonl")
    parser.add_argument("--base-url", default=os.environ.get(
        "MLX_OMARCHY_COMMUNITY_URL", DEFAULT_BASE_URL))
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--kind", default=None,
                        help="substring: quick, deep, ...")
    parser.add_argument("--chip", default=None,
                        help="substring: Apple M1, M2, ...")
    parser.add_argument("--kernel", default=None,
                        help="substring of the kernel release")
    parser.add_argument("--cpu-present", type=int, default=None,
                        help="exact count of present cores")
    parser.add_argument("--hotplug", choices=("yes", "no"), default=None,
                        help="per-cpu hotplug control (cpuN/online) present")
    parser.add_argument("--ane", choices=("yes", "no"), default=None,
                        help="ANE node in the booted devicetree")
    parser.add_argument("--shortfall", choices=("yes", "no"), default=None,
                        help="running fewer cores than present, unexplained")
    parser.add_argument("--boot", default=None,
                        help="substring of the boot chain: m1n1, iBoot, fw")
    parser.add_argument("--cmdline", default=None,
                        help="substring of the kernel command line")
    parser.add_argument("--mesa", default=None,
                        help="substring of driver name, API or package")
    parser.add_argument("--mlx-version", default=None,
                        help="substring of the mlx-omarchy version")
    parser.add_argument("--metric", default="tflops",
                        help="compare: tflops, median_ms, min_ms, max_ms")
    parser.add_argument("--size", type=int, default=None,
                        help="compare: matmul matrix size n")
    parser.add_argument("command", nargs="?", default="list",
                        choices=("list", "show", "compare"))
    parser.add_argument("target", nargs="?", default=None,
                        help="show: sha256 prefix")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    try:
        if args.command == "show" and not args.target:
            print("show needs a sha256 prefix.", file=sys.stderr)
            return 1
        records, skipped, source = load_dataset(args, repo_root)
        records = filter_records(records, args)
        if args.command == "list":
            return cmd_list(records, skipped, source, args.json)
        if args.command == "show":
            return cmd_show(records, args.target, args)
        return cmd_compare(records, args.metric, args.size, args.json)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
