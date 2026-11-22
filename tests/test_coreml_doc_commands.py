# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Every repo-relative path in bash examples of docs/coreml.md must be git-tracked."""

import posixpath
import shlex
from pathlib import Path

import pytest

pytest.importorskip(
    "markdown_it",
    reason="markdown-it-py is not installed (pip install -r requirements-dev.txt)",
)

from markdown_it import MarkdownIt

from tests.test_doc_relative_links import _git_tracked_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_REL = "docs/coreml.md"


def find_missing_command_paths(text: str) -> list[str]:
    files, dirs = _git_tracked_paths(REPO_ROOT)
    top_dirs = {p.split("/", 1)[0] for p in files if "/" in p}
    missing: list[str] = []
    for tok in MarkdownIt("commonmark").parse(text):
        if tok.type != "fence" or tok.info.strip() != "bash":
            continue
        for arg in shlex.split(tok.content.replace("\\\n", " "), comments=True):
            value = arg.split("=", 1)[-1] if arg.startswith("-") else arg
            norm = posixpath.normpath(value)
            if "/" in norm and norm.split("/", 1)[0] in top_dirs:
                if norm not in files and norm not in dirs:
                    missing.append(f"{DOC_REL}:{tok.map[0] + 1} -> {value}")
    return missing


def test_coreml_doc_bash_command_paths_are_tracked() -> None:
    missing = find_missing_command_paths((REPO_ROOT / DOC_REL).read_text(encoding="utf-8"))
    assert not missing, "Missing repo paths in bash examples:\n" + "\n".join(missing)


def test_checker_flags_missing_repo_paths_only() -> None:
    text = (
        "```bash\n"
        "cmd docs/coreml.md receipts/missing.json ./docs/gone.md \\\n"
        "  --opt=tests/gone.py /path/to/report.json PATH # docs/comment.md\n"
        "```\n"
    )
    assert find_missing_command_paths(text) == [
        f"{DOC_REL}:1 -> receipts/missing.json",
        f"{DOC_REL}:1 -> ./docs/gone.md",
        f"{DOC_REL}:1 -> tests/gone.py",
    ]
