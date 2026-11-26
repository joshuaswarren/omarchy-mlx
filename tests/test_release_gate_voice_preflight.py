import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

GATES = Path(__file__).resolve().parents[1] / "scripts" / "release-gates"
VOICE_MODULES = (
    "mlx_audio",
    "misaki",
    "num2words",
    "spacy",
    "en_core_web_sm",
    "phonemizer",
    "espeakng_loader",
)
MISSING = "mlx_audio"


def make_gate_home(root, present):
    """A gate home whose venv python imports only the modules in `present`."""
    home = root / "gate-home"
    venv_bin = home / ".local" / "share" / "mlx-omarchy" / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.write_text(
        "#!/bin/sh\n"
        f'case " {" ".join(present)} " in\n'
        '  *" ${2#import } "*) exit 0 ;;\n'
        "esac\n"
        "exit 1\n"
    )
    python.chmod(0o755)
    return home


def gate_env(root, home):
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(root),
        "TAG": "v0-voice-preflight",
        "GATE_ROOT": str(root),
        "GATE_HOME": str(home),
        "LOG_DIR": str(root / "logs"),
    }


def helper_run(root, home):
    return subprocess.run(
        ["bash", "-c", '. "$1" && gate_require_voice_runtime "$2"', "_",
         str(GATES / "env.sh"), str(root / "gate.log")],
        env=gate_env(root, home),
        capture_output=True,
        text=True,
        timeout=60,
    )


class VoiceRuntimePreflightTests(unittest.TestCase):
    def run_gate(self, script):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            present = [name for name in VOICE_MODULES if name != MISSING]
            home = make_gate_home(root, present)
            started = time.monotonic()
            proc = subprocess.run(
                ["bash", str(GATES / script)],
                env=gate_env(root, home),
                capture_output=True,
                text=True,
                timeout=120,
            )
            elapsed = time.monotonic() - started
            server_log = root / "logs" / f"{Path(script).stem}.log.server"
            return proc, elapsed, server_log.exists()

    def assert_refused_by_name(self, script):
        proc, elapsed, server_started = self.run_gate(script)
        output = proc.stdout + proc.stderr
        refusal = [line for line in output.splitlines() if "voice runtime missing" in line]
        self.assertNotEqual(proc.returncode, 0, output)
        self.assertEqual(len(refusal), 1, output)
        self.assertIn(MISSING, refusal[0])
        self.assertNotIn("misaki", refusal[0])
        self.assertLess(elapsed, 60, "must refuse before the primer timeout")
        self.assertFalse(server_started, "server must not start when the runtime is missing")

    def test_g10_refuses_and_names_the_missing_voice_package(self):
        self.assert_refused_by_name("g10-kokoro-primer.sh")

    def test_g12_refuses_and_names_the_missing_voice_package(self):
        self.assert_refused_by_name("g12-kokoro-stream.sh")

    def test_helper_is_silent_when_every_voice_package_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = helper_run(root, make_gate_home(root, list(VOICE_MODULES)))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(proc.stdout + proc.stderr, "")

    def test_helper_names_only_the_missing_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            present = [name for name in VOICE_MODULES if name != MISSING]
            proc = helper_run(root, make_gate_home(root, present))
            output = proc.stdout + proc.stderr
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(MISSING, output)
            self.assertNotIn("misaki", output)


if __name__ == "__main__":
    unittest.main()
