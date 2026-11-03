"""Fixture test for ab.py's failure contract; needs no GPU, flock or
journalctl. A child that exits 3 (provenance refusal) or 124 (timeout)
must end the run with that status after its record is written, and no
later arm may run. With GPU error monitoring unavailable (no journalctl
on PATH, or a journalctl that exits non-zero) ab.py must refuse with a
non-zero status before any arm runs and write no record.

usage: python3 test_ab.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def write(path, text, mode=0o755):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, mode)


def setup(root, child_rc, journalctl):
    write(f"{root}/commits.txt", "a 1111111\nb 2222222\n", 0o644)
    os.makedirs(f"{root}/probes")
    os.makedirs(f"{root}/bin")
    write(f"{root}/bin/timeout", '#!/bin/sh\nshift\nexec "$@"\n')
    if journalctl is not None:
        write(f"{root}/bin/journalctl", journalctl)
    for arm in ("a", "b"):
        os.makedirs(f"{root}/venv-{arm}/bin")
        write(f"{root}/venv-{arm}/bin/python",
              f'#!/bin/sh\necho {arm} >> "{root}/ran.txt"\nexit {child_rc}\n')
    # PATH holds only the stubs, so a host journalctl cannot stand in.
    env = dict(os.environ, PATH=f"{root}/bin", AB_ROOT=root, AB_NOWAIT="1", AB_NO_LOCK="1")
    out = f"{root}/out.jsonl"
    p = subprocess.run([sys.executable, f"{HERE}/ab.py", out, "2", "a", "b", "--", "probe.py"],
                       env=env, capture_output=True, text=True)
    return p, out


def run_case(child_rc):
    with tempfile.TemporaryDirectory() as root:
        p, out = setup(root, child_rc, "#!/bin/sh\nexit 0\n")
        recs = [json.loads(line) for line in open(out)]
        ran = open(f"{root}/ran.txt").read().split()
        assert p.returncode == child_rc, (child_rc, p.returncode, p.stdout, p.stderr)
        assert [r["rc"] for r in recs] == [child_rc], recs
        assert ran == ["a"], ran
        assert "STOP" in p.stdout, p.stdout
    print(f"child rc {child_rc}: ab.py exit {child_rc}, 1 record kept, later arms not run")


def monitor_case(label, journalctl):
    with tempfile.TemporaryDirectory() as root:
        p, out = setup(root, 0, journalctl)
        assert p.returncode != 0, (label, p.returncode, p.stdout, p.stderr)
        assert "REFUSED: GPU error monitoring is unavailable" in p.stderr, (label, p.stderr)
        assert not os.path.exists(out) or os.path.getsize(out) == 0, label
        assert not os.path.exists(f"{root}/ran.txt"), label
    print(f"monitor {label}: refused with exit {p.returncode}, no record, no arm run")


def main():
    for rc in (3, 124):
        run_case(rc)
    monitor_case("absent (no journalctl on PATH)", None)
    monitor_case("failing (journalctl exits 1)", "#!/bin/sh\necho denied >&2\nexit 1\n")
    print("test_ab: OK")


if __name__ == "__main__":
    main()
