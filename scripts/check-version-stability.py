#!/usr/bin/env python3
"""Check the prepared setup.py version across separate build-hook clocks."""
import ast
import datetime
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

source = Path(sys.argv[1]).resolve()
setup_path = source / "setup.py"
function = next(node for node in ast.parse(setup_path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "get_version")
code = compile(ast.Module(body=[function], type_ignores=[]), str(setup_path), "exec")
os.chdir(source)
os.environ.update(DEV_RELEASE="1", PYPI_RELEASE="0", MLX_OMARCHY_LOCAL_VERSION="check")
os.environ.pop("SOURCE_DATE_EPOCH", None)


def version_at(clock):
    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(clock, tz)

    namespace = {"datetime": SimpleNamespace(datetime=Clock, timezone=datetime.timezone),
                 "os": os, "subprocess": subprocess, "__file__": str(setup_path)}
    exec(code, namespace)
    return namespace["get_version"]()


first = version_at(1704067199)
second = version_at(1704153601)
assert first == second, (first, second)
os.environ["SOURCE_DATE_EPOCH"] = "1704067200"
assert version_at(1704153601).endswith(".dev202401010000+check")
print("PASS: build-hook clocks cannot change version; SOURCE_DATE_EPOCH uses UTC")
