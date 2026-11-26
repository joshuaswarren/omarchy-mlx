"""provrec.py <tree> <hf_repo> [<icd_or_empty>] : print one JSON object with the provenance fields of a scoreboard run.

Reads only. Does not touch the GPU. The tree is a wheel unpacked at <tree> (mlx/, mlx_lm/, *.dist-info).
Fields: kernel, wheel version, libmlx and extension sha256 vs the wheel RECORD, model snapshot commit, per-file
sha256 (HF LFS blobs are named by their sha256), quantization from config.json, Vulkan device/driver identity,
GPU firmware lines, governor, boot-id prefix.
"""
import base64
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

tree, repo = sys.argv[1], sys.argv[2]
out = {"tree": os.path.basename(tree.rstrip("/"))}


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    return p.stdout.strip()


out["kernel"] = os.uname().release
out["boot_id8"] = open("/proc/sys/kernel/random/boot_id").read()[:8]
out["governor"] = open("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor").read().strip()

dist = glob.glob(os.path.join(tree, "mlx_omarchy-*.dist-info"))
out["wheel"] = os.path.basename(dist[0])[len("mlx_omarchy-"):-len(".dist-info")] if dist else "missing"
rec = {}
if dist:
    for line in open(os.path.join(dist[0], "RECORD")):
        p = line.rstrip("\n").split(",")
        if len(p) == 3 and p[1].startswith("sha256="):
            rec[p[0]] = p[1][7:]
files = {}
for pat in ("mlx/lib/libmlx.so", "mlx/core*.so"):
    for f in glob.glob(os.path.join(tree, pat)):
        h = hashlib.sha256(open(f, "rb").read()).digest()
        rel = os.path.relpath(f, tree)
        b64 = base64.urlsafe_b64encode(h).decode().rstrip("=")
        files[rel] = {"sha256": h.hex(), "matches_wheel_record": rec.get(rel) == b64}
out["wheel_files"] = files

hub = os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--"))
snaps = sorted(os.listdir(os.path.join(hub, "snapshots"))) if os.path.isdir(hub) else []
out["model_repo"] = repo
out["model_snapshot"] = snaps[-1] if snaps else "missing"
if snaps:
    sd = os.path.join(hub, "snapshots", snaps[-1])
    blobs = {}
    for f in sorted(os.listdir(sd)):
        target = os.path.basename(os.path.realpath(os.path.join(sd, f)))
        if f.endswith(".safetensors") or f.endswith(".safetensors.index.json") or f == "config.json":
            blobs[f] = target
    out["model_files"] = blobs
    cfg = json.load(open(os.path.join(sd, "config.json")))
    out["quantization"] = cfg.get("quantization") or cfg.get("quantization_config")
    out["model_type"] = cfg.get("model_type")

vk = sh("vulkaninfo --summary 2>/dev/null")
dev = {}
for key in ("deviceName", "driverID", "driverName", "driverInfo", "apiVersion", "deviceType"):
    m = re.search(r"^\s*" + key + r"\s*=\s*(.+)$", vk, re.M)
    if m:
        dev[key] = m.group(1).strip()
out["vulkan"] = dev
out["driver_lib_sha256_12"] = sh("sha256sum /usr/lib/libvulkan_asahi.so 2>/dev/null | cut -c1-12")
out["driver_package"] = sh("pacman -Q vulkan-asahi-omarchy 2>/dev/null")
kj = sh("journalctl -k -b --no-pager 2>/dev/null | grep -E 'gpu: RTKit: Initializing|Initialized .* [0-9.]+ for .*gpu on minor'")
m = re.search(r"Initialized \S+ ([0-9.]+) for", kj)
out["gpu_kernel_driver_version"] = m.group(1) if m else "not in journal"
m = re.search(r"gpu: RTKit: Initializing \(protocol version (\d+)\)", kj)
out["gpu_rtkit_protocol"] = m.group(1) if m else "not in journal"
print(json.dumps(out, indent=1))
