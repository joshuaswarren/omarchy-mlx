"""The site fit-table export must agree with budget.py, never re-derive it."""

import contextlib
import errno
import io
import json
import os
import runpy
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import budget, catalog  # noqa: E402
from mlx_omarchy_serve.catalog import CatalogError, bundled_path  # noqa: E402
from mlx_omarchy_serve.export_fit_table import MEMORY_SIZES_GIB, build_fit_table, main  # noqa: E402


def test_export_cells_match_budget_calls():
    table = build_fit_table()
    assert table["memory_sizes_gib"] == MEMORY_SIZES_GIB
    cat_models = {m["id"]: m for m in json.loads(
        bundled_path().read_text(encoding="utf-8"))["models"]}
    for row in table["models"]:
        entry = cat_models[row["id"]]
        default_ctx = budget.resolve_context(entry["context"], None)
        for gib in MEMORY_SIZES_GIB:
            cell = row["fits"][str(gib)]
            available = gib * budget.GiB - budget.SAFETY_RESERVE_BYTES
            fits_at = budget.estimate_required(entry["memory"], cell["max_context"]).total
            assert fits_at <= available or cell["max_context"] == 0
            one_over = budget.estimate_required(entry["memory"], cell["max_context"] + 1).total
            assert one_over > available or cell["max_context"] == entry["context"]["max_tokens"]
            expected_fits = (
                cell["max_context"] >= default_ctx
                and (entry["capability"]["min_mem_gib"] is None
                     or entry["capability"]["min_mem_gib"] <= gib)
            )
            assert cell["fits"] is expected_fits, (row["id"], gib)


def test_export_is_json_serializable_and_sanitized():
    table = build_fit_table()
    text = json.dumps(table)
    for private in ("jwm1", "jw16", "jw14m2", "macstudio", "192.168.", "100."):
        assert private not in text


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "directory",
        pytest.param(
            "unreadable",
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod 0"),
        ),
        "oversized",
        "non_utf8",
        "malformed_json",
        "int_digit_limit",
    ],
)
def test_build_fit_table_rejects_unreadable_and_malformed_catalog(tmp_path, case):
    path = tmp_path / f"{case}.json"
    if case == "directory":
        path.mkdir()
    elif case == "unreadable":
        path.write_bytes(b"{}")
        path.chmod(0)
    elif case == "oversized":
        raw = bundled_path().read_bytes()
        path.write_bytes(raw + b" " * (catalog.CATALOG_MAX_BYTES + 1 - len(raw)))
    elif case == "non_utf8":
        path.write_bytes(b'{"version": "\xff"}')
    elif case == "malformed_json":
        path.write_bytes(b'{"version": 1,')
    elif case == "int_digit_limit":
        path.write_bytes(b'{"version": ' + b"9" * 5000 + b"}")

    prev_int_digits: int | None = None
    try:
        if case == "int_digit_limit":
            prev_int_digits = sys.get_int_max_str_digits()
            sys.set_int_max_str_digits(4300)
        with pytest.raises(CatalogError) as exc_info:
            build_fit_table(path)
        msg = str(exc_info.value)
        assert str(path) in msg
        if case == "missing":
            assert msg.count(str(path)) == 1
    finally:
        if prev_int_digits is not None:
            sys.set_int_max_str_digits(prev_int_digits)
        if case == "unreadable":
            path.chmod(0o600)


def test_build_fit_table_rejects_deeply_nested_json_pinned_stack(tmp_path):
    deep = "[" * 120_000 + "]" * 120_000
    path = tmp_path / "deep.json"
    path.write_text(deep, encoding="utf-8")

    caught: list[BaseException] = []

    def worker() -> None:
        try:
            with pytest.raises(RecursionError):
                json.loads(deep)
            with pytest.raises(CatalogError):
                build_fit_table(path)
        except BaseException as exc:
            caught.append(exc)

    prev = threading.stack_size(4 * 1024 * 1024)
    try:
        thread = threading.Thread(target=worker)
        thread.start()
    finally:
        threading.stack_size(prev)
    thread.join()
    if caught:
        raise caught[0]


def test_build_fit_table_rejects_json_recursion_error(tmp_path, monkeypatch):
    path = tmp_path / "small.json"
    path.write_text("{}", encoding="utf-8")

    def boom(_raw):
        raise RecursionError("simulated nesting overflow")

    monkeypatch.setattr(catalog.json, "loads", boom)
    with pytest.raises(CatalogError):
        build_fit_table(path)


@pytest.mark.parametrize("case", ["missing", "malformed_json", "schema_invalid"])
def test_main_reports_error_on_bad_catalog(tmp_path, capsys, case):
    path = tmp_path / f"{case}.json"
    if case == "malformed_json":
        path.write_bytes(b'{"version": 1,')
    elif case == "schema_invalid":
        path.write_text('{"version": 2}', encoding="utf-8")

    rc = main(["--catalog", str(path)])
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert rc == 1
    assert captured.out == ""
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("out_rel", ["adir", "no/such/out.json"])
def test_main_reports_error_on_unwritable_out(tmp_path, capsys, out_rel):
    target = tmp_path / out_rel
    if out_rel == "adir":
        target.mkdir()

    rc = main(["-o", str(target)])
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert rc == 1
    assert captured.out == ""
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err


def test_main_writes_out_matching_stdout(tmp_path, capsys):
    rc_stdout = main([])
    stdout_cap = capsys.readouterr()
    assert rc_stdout == 0
    assert stdout_cap.err == ""

    utf16_catalog = tmp_path / "catalog-utf16.json"
    utf16_catalog.write_bytes(bundled_path().read_text(encoding="utf-8").encode("utf-16"))
    out_path = tmp_path / "fit-table.json"
    rc_out = main(["--catalog", str(utf16_catalog), "-o", str(out_path)])
    out_cap = capsys.readouterr()
    assert rc_out == 0
    assert out_cap.out == ""
    assert out_cap.err == ""
    assert out_path.read_bytes() == stdout_cap.out.encode("utf-8")


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="/dev/full not available")
def test_main_in_process_full_stdout_preserves_fd(capsys):
    full_rdev = os.stat("/dev/full").st_rdev
    full = open("/dev/full", "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(full):
            rc = main([])
        assert os.fstat(full.fileno()).st_rdev == full_rdev
    finally:
        with contextlib.suppress(OSError):
            full.close()
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert rc == 1
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert os.strerror(errno.ENOSPC) in lines[0]


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="/dev/full not available")
@pytest.mark.parametrize("catalog_kind", ["bundled", "small"])
def test_main_reports_error_on_full_stdout(tmp_path, catalog_kind):
    cmd = [sys.executable, "-m", "mlx_omarchy_serve.export_fit_table"]
    if catalog_kind == "small":
        cat = json.loads(bundled_path().read_text(encoding="utf-8"))
        cat["models"] = cat["models"][:1]
        cat.pop("pairs", None)
        small = tmp_path / "small.json"
        small.write_text(json.dumps(cat), encoding="utf-8")
        assert len(json.dumps(build_fit_table(small), indent=2)) < 8192
        cmd.extend(["--catalog", str(small)])
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    env.pop("PYTHONUNBUFFERED", None)
    with open("/dev/full", "wb") as full:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT / "serve",
            env=env,
            stdout=full,
            stderr=subprocess.PIPE,
            text=True,
        )
    lines = proc.stderr.splitlines()
    assert proc.returncode == 1
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Exception ignored" not in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="/bin/sh not available")
@pytest.mark.parametrize("case", ["missing_catalog", "valid_stdout", "valid_out"])
def test_main_reports_error_when_stdout_closed_at_launch(tmp_path, capsys, case):
    missing = tmp_path / "missing.json"
    out_path = tmp_path / "out.json"
    if case == "missing_catalog":
        cli_args = ["--catalog", str(missing)]
    elif case == "valid_out":
        cli_args = ["-o", str(out_path)]
    else:
        cli_args = []
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    proc = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'exec "$0" "$@" >&-',
            sys.executable,
            "-m",
            "mlx_omarchy_serve.export_fit_table",
            *cli_args,
        ],
        cwd=REPO_ROOT / "serve",
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    if case == "valid_out":
        assert proc.returncode == 0
        assert proc.stderr == ""
        assert main([]) == 0
        expected = capsys.readouterr()
        assert expected.err == ""
        assert out_path.read_bytes() == expected.out.encode("utf-8")
        return
    lines = proc.stderr.splitlines()
    assert proc.returncode == 1
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in proc.stderr
    if case == "missing_catalog":
        assert str(missing) in lines[0]
    else:
        assert "stdout" in lines[0].lower()


def test_main_module_keeps_healthy_stdout_on_error(tmp_path, monkeypatch, capsys):
    buf = io.StringIO()
    monkeypatch.delitem(sys.modules, "mlx_omarchy_serve.export_fit_table", raising=False)
    monkeypatch.setattr(sys, "argv", ["export_fit_table", "--catalog", str(tmp_path / "missing.json")])
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc_info:
        runpy.run_module("mlx_omarchy_serve.export_fit_table", run_name="__main__")
    assert exc_info.value.code == 1
    buf.write("still open\n")
    assert buf.getvalue() == "still open\n"
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err
