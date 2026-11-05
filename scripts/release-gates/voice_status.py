"""Explain a voice synthesis status that did not prime (used by the g10 driver).

`voice` is the `voice.synthesis` object from GET /api/status. A dependency that
imports reports its version and its pin; `satisfies` is False when the installed
version breaks the pin, and that case leaves `missing` empty.
"""


def primer_failure_line(voice):
    deps = voice.get("dependencies") or {}
    conflicts = [
        f"{name} {info['version']} violates {info['constraint']}"
        for name, info in (deps.get("detail") or {}).items()
        if isinstance(info, dict) and info.get("satisfies") is False
    ]
    return (f"PRIMER_NOT_LANDED usable={voice.get('usable')} state={voice.get('state')} "
            f"dependencies.missing={deps.get('missing')} conflicts={conflicts}")
