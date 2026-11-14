import json
import os


def load_config(path):
    with open(path) as f:
        data = json.load(f)
    if "name" not in data:
        data["name"] = os.path.basename(path)
    return data


def merge_configs(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def find_files(root, suffix):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(suffix):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def summarize(configs):
    names = [c["name"] for c in configs]
    total = sum(len(c) for c in configs)
    return {"count": len(configs), "names": names, "keys": total}


def main(root):
    files = find_files(root, ".json")
    configs = [load_config(p) for p in files]
    base = configs[0] if configs else {}
    merged = base
    for c in configs[1:]:
        merged = merge_configs(merged, c)
    print(json.dumps(summarize(configs), indent=2))
    print(json.dumps(merged, indent=2))
