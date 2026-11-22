# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Every relative Markdown link outside receipt folders must resolve in the repo."""

import posixpath
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from markdown_it import MarkdownIt
from markdown_it.token import Token

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_TOP_DIRS = frozenset({"receipts", "receipts-work"})
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _drop_footnote_refs(state) -> None:
    refs = state.env.get("references")
    if refs:
        for key in [k for k in refs if k.startswith("^")]:
            del refs[key]


_MD = MarkdownIt("commonmark").enable("table")
_MD.core.ruler.before("inline", "drop_footnote_refs", _drop_footnote_refs)


def _walk_inline_tokens(
    tokens: list[Token], lineno: int, found: list[tuple[int, str]]
) -> int:
    for tok in tokens:
        if tok.type in ("softbreak", "hardbreak"):
            lineno += 1
        elif tok.type == "html_inline":
            lineno += tok.content.count("\n")
        elif tok.type == "link_open":
            href = tok.attrGet("href")
            if isinstance(href, str):
                found.append((lineno, href))
        elif tok.type == "image":
            src = tok.attrGet("src")
            if isinstance(src, str):
                found.append((lineno, src))
        if tok.children:
            # An image's children are its alt text, which renders as plain text.
            sink = [] if tok.type == "image" else found
            lineno = _walk_inline_tokens(tok.children, lineno, sink)
    return lineno


def _iter_relative_links(text: str):
    found: list[tuple[int, str]] = []
    current_line = 1
    for tok in _MD.parse(text):
        if tok.map:
            current_line = tok.map[0] + 1
        if tok.type == "inline" and tok.children:
            _walk_inline_tokens(tok.children, current_line, found)

    for lineno, target in found:
        if (
            not target
            or target.startswith("#")
            or target.startswith("//")
            or _SCHEME_RE.match(target)
        ):
            continue
        path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if path_part:
            yield lineno, target, path_part


def _git_tracked_paths(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
    files = frozenset(
        entry.decode("utf-8", errors="surrogateescape") for entry in raw.split(b"\0") if entry
    )
    dirs: set[str] = {"."}
    for rel in files:
        parent = posixpath.dirname(rel)
        while parent:
            dirs.add(parent)
            parent = posixpath.dirname(parent)
    return files, frozenset(dirs)


def _normalize_repo_target(source_rel: str, path_part: str) -> str | None:
    if path_part.startswith("/"):
        joined = path_part.lstrip("/") or "."
    else:
        source_dir = posixpath.dirname(source_rel)
        joined = posixpath.join(source_dir, path_part) if source_dir else path_part
    normalized = posixpath.normpath(joined)
    if normalized == ".." or normalized.startswith("../"):
        return None
    return normalized


def _target_is_tracked(
    source_rel: str,
    path_part: str,
    tracked_files: frozenset[str],
    tracked_dirs: frozenset[str],
) -> bool:
    normalized = _normalize_repo_target(source_rel, path_part)
    if normalized is None:
        return False
    if path_part.endswith("/"):
        return normalized in tracked_dirs
    return normalized in tracked_files or normalized in tracked_dirs


def find_dead_relative_links(repo_root: Path) -> list[str]:
    root = repo_root.resolve()
    tracked_files, tracked_dirs = _git_tracked_paths(root)
    dead: list[str] = []
    for rel in sorted(tracked_files):
        if not rel.endswith(".md") or rel.split("/", 1)[0] in SKIP_TOP_DIRS:
            continue
        md = root / rel
        if not md.is_file():
            continue
        for lineno, target, path_part in _iter_relative_links(md.read_text(encoding="utf-8")):
            if not _target_is_tracked(rel, path_part, tracked_files, tracked_dirs):
                dead.append(f"{rel}:{lineno} -> {target}")
    return dead


def _git_init(root: Path, *paths: str) -> None:
    subprocess.run(["git", "-C", str(root), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", *paths], check=True, capture_output=True)


def test_non_receipt_markdown_has_no_dead_relative_links() -> None:
    dead = find_dead_relative_links(REPO_ROOT)
    assert not dead, "Dead relative Markdown links:\n" + "\n".join(dead)


def test_checker_catches_dead_links_and_ignores_non_links(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Root\n\n[guide](README.md&#35;usage)\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "real.md").write_text("# Real\n", encoding="utf-8")
    (tmp_path / "docs" / "dir(a(b)).md").write_text("# Nested\n", encoding="utf-8")
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "old.md").write_text("[ignored](missing-in-receipts.md)\n", encoding="utf-8")
    (tmp_path / "docs" / "sample.md").write_text(
        "[ok](real.md#section) [root-ok](/README.md) [root-dir](/docs/) [nested](dir(a(b)).md)\n"
        "[ext](https://example.com/x) [proto](//example.com/x) [anchor](#local)\n"
        "`[code](missing-code.md)` \\[escaped](missing.md) <!-- [example](missing.md) -->\n"
        "See[^1].\n\n[^1]: word\n\n"
        "Paragraph text\n[Note]: missing.md\n\n"
        "| col |\n| --- |\n| [table-ok](real.md) [table-dead](missing-table.md) |\n\n"
        "```md\n[fenced](missing-fenced.md)\n```\n\n"
        "    [indented](missing-indented.md)\n\n"
        "> [guide](\n> missing.md)\n\n"
        "[report][r]\n"
        "[![badge](missing-badge.png)](real.md)\n"
        "[escape](../../outside.md)\n"
        "[root-dead](/missing)\n"
        "[root-escape](/../outside.md)\n"
        "[file-as-dir](real.md/)\n"
        "![See [alt](missing-alt.md)](real.md)\n\n"
        '[r]: missing-ref.md "line one\nline two"\n',
        encoding="utf-8",
    )
    _git_init(tmp_path, ".")
    assert find_dead_relative_links(tmp_path) == [
        "docs/sample.md:13 -> missing-table.md",
        "docs/sample.md:21 -> missing.md",
        "docs/sample.md:24 -> missing-ref.md",
        "docs/sample.md:25 -> missing-badge.png",
        "docs/sample.md:26 -> ../../outside.md",
        "docs/sample.md:27 -> /missing",
        "docs/sample.md:28 -> /../outside.md",
        "docs/sample.md:29 -> real.md/",
    ]


def test_git_repo_checks_only_tracked_files_and_targets(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "tracked.md").write_text("# Tracked\n", encoding="utf-8")
    (tmp_path / "docs" / "untracked-target.md").write_text("# Untracked\n", encoding="utf-8")
    (tmp_path / "untracked-root.md").write_text("[ignored](missing-root.md)\n", encoding="utf-8")
    nested_nm = tmp_path / "services" / "community-data" / "node_modules" / "pkg"
    nested_nm.mkdir(parents=True)
    (nested_nm / "README.md").write_text("[ignored-nm](missing-nm.md)\n", encoding="utf-8")
    (tmp_path / "docs" / "index.md").write_text(
        "[ok](tracked.md) [dir](./) [root](/README.md) [root-dir](/docs/)\n"
        "[bad-untracked](untracked-target.md)\n"
        "[bad-escape](../../outside.md)\n"
        "[root-dead](/missing)\n"
        "[root-escape](/../outside.md)\n"
        "[file-as-dir](tracked.md/)\n",
        encoding="utf-8",
    )
    _git_init(tmp_path, "README.md", "docs/tracked.md", "docs/index.md")
    assert find_dead_relative_links(tmp_path) == [
        "docs/index.md:2 -> untracked-target.md",
        "docs/index.md:3 -> ../../outside.md",
        "docs/index.md:4 -> /missing",
        "docs/index.md:5 -> /../outside.md",
        "docs/index.md:6 -> tracked.md/",
    ]
