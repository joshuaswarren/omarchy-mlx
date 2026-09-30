# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""The one name table for mlx-omarchy installs, plus venv discovery.

Every package, path, launcher, unit, and desktop-entry name that the
installer, the serve CLI, the assistant, and the retire tooling share
is defined here once. ``packaging/paths.sh`` is generated output
(``python3 serve/mlx_omarchy_paths.py --shell``); a contract test
regenerates it and fails on drift. The planned repo/package rename to
omarchy-mlx is an edit to this file and its generated outputs only.

Venv discovery precedence, used by every launcher, the installer, the
serve CLI, and the assistant:

1. ``$OMARCHY_MLX_VENV`` — explicit override, wins when it exists.
2. ``/usr/lib/omarchy-mlx/venv`` — the system package layout. The
   prefix is a constant; callers may pass another prefix (tests).
3. ``~/.local/share/mlx-omarchy/venv`` — the legacy home install.
   Resolving here prints one stderr line naming the retire command, so
   a leftover home install announces itself once the system package
   exists.

User data never lives inside a venv: chat history, pair homes, and
serve state sit under the data root (``$MLX_OMARCHY_HOME``, default
``~/.local/share/mlx-omarchy``) and model weights in
``~/.cache/huggingface``. Both survive venv replacement.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = "joshuawarren/mlx-omarchy"
HOME_PREFIX_NAME = "mlx-omarchy"
SYSTEM_PREFIX = "/usr/lib/omarchy-mlx"
SYSTEM_SHARE_PREFIX = "/usr/share/omarchy-mlx"
VENV_DIR_NAME = "venv"
DATA_HOME_ENV = "MLX_OMARCHY_HOME"
VENV_ENV = "OMARCHY_MLX_VENV"
PYTHON_VERSION = "3.14"

RETIRE_CMD = "mlx-omarchy-retire-legacy"
HOME_LAUNCHERS = (
    "mlx-omarchy",
    "mlx-omarchy-demo",
    "mlx-omarchy-chat",
    "mlx-omarchy-info",
    "mlx-omarchy-serve",
    "omarchy-mlx-serve",
)
UNIT_NAME = "mlx-omarchy-chat.service"
DESKTOP_NAMES = ("mlx-omarchy-chat.desktop",)
SERVE_PACKAGES = (
    "mlx_omarchy_serve",
    "mlx_omarchy_laya",
    "mlx_omarchy_bonsai2",
    "mlx_omarchy_assistant",
)


def home_prefix(home: Path | None = None) -> Path:
    """The legacy install root and user-data root ($MLX_OMARCHY_HOME)."""
    override = os.environ.get(DATA_HOME_ENV, "")
    if override:
        return Path(override)
    return (Path(home) if home else Path.home()) / ".local" / "share" / HOME_PREFIX_NAME


def default_data_home() -> Path:
    """The user-data root: serve state, chat history, pair homes."""
    return home_prefix()


def system_venv(prefix: str | Path | None = None) -> Path:
    """The packaged venv; pass ``prefix`` to stage or test elsewhere."""
    return Path(prefix) / VENV_DIR_NAME if prefix else Path(SYSTEM_PREFIX) / VENV_DIR_NAME


def legacy_venv(home: Path | None = None) -> Path:
    """The pre-package home install venv."""
    return home_prefix(home) / VENV_DIR_NAME


def venv_roots(
    *, system_prefix: str | Path | None = None, home: Path | None = None
) -> list[Path]:
    """Existing venv roots in precedence order: override, system, legacy.

    Glob-style consumers (the recognition tools tree, the Parakeet CLI
    lookup) search every existing root so an override or a system venv
    never hides a second install.
    """
    candidates = []
    override = os.environ.get(VENV_ENV, "")
    if override:
        candidates.append(Path(override))
    candidates.append(system_venv(system_prefix))
    candidates.append(legacy_venv(home))
    return [root for root in candidates if root.is_dir()]


def discover_venv(
    *, system_prefix: str | Path | None = None, home: Path | None = None
) -> tuple[Path | None, str]:
    """First existing venv root with its origin ("env", "system", "legacy").

    The legacy home venv is the last resort; resolving there prints the
    one-line retire hint on stderr so a leftover home install announces
    itself once the system package exists.
    """
    ordered = [(system_venv(system_prefix), "system"), (legacy_venv(home), "legacy")]
    override = os.environ.get(VENV_ENV, "")
    if override:
        ordered.insert(0, (Path(override), "env"))
    for candidate, origin in ordered:
        if candidate.is_dir():
            if origin == "legacy":
                print(
                    f"warning: {candidate} is the legacy home venv; the system "
                    f"venv is preferred - run {RETIRE_CMD} to remove the legacy install",
                    file=sys.stderr,
                )
            return candidate, origin
    return None, "none"


# packaging/paths.sh is generated from this table; do not edit that file.
SHELL_TABLE = (
    ("REPO", REPO),
    ("HOME_NAME", HOME_PREFIX_NAME),
    ("SYSTEM_PREFIX", SYSTEM_PREFIX),
    ("SYSTEM_SHARE_PREFIX", SYSTEM_SHARE_PREFIX),
    ("VENV_DIR", VENV_DIR_NAME),
    ("DATA_HOME_ENV", DATA_HOME_ENV),
    ("VENV_ENV", VENV_ENV),
    ("PYTHON_VERSION", PYTHON_VERSION),
    ("RETIRE_CMD", RETIRE_CMD),
    ("LAUNCHERS", " ".join(HOME_LAUNCHERS)),
    ("UNIT_NAME", UNIT_NAME),
    ("DESKTOPS", " ".join(DESKTOP_NAMES)),
    ("SERVE_PACKAGES", " ".join(SERVE_PACKAGES)),
)


def shell_script() -> str:
    """The bash projection: a strict KEY="value" file any script can source."""
    lines = [
        "# Generated by serve/mlx_omarchy_paths.py --shell; edit that file instead.",
        "# Strict KEY=\"value\" lines only: also parsed by tests as name truth.",
    ]
    lines.extend(f'{key}="{value}"' for key, value in SHELL_TABLE)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--shell"]:
        sys.stdout.write(shell_script())
        return 0
    print("usage: python3 mlx_omarchy_paths.py --shell", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
