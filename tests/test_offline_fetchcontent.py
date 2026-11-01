"""Real CMake configure boundary with generated sources, no compilation or network.

Uses the builder's actual fixed CMAKE_ARGS assignment; source fixtures are not
native dependency identities, and this does not execute the wheel builder.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'packaging/offline-fetchcontent.cmake'
DEPS = ('JSON', 'FMT', 'NANOBIND', 'GGUFLIB')


class OfflineFetchContentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mlx offline ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'project'
        self.source.mkdir()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.tripwire = self.root / 'download-attempt'
        git = self.bin / 'git'
        git.write_text('#!/bin/sh\nif [ "$1" = --version ]; then echo "git version 2.55.0"; exit 0; fi\nprintf "%s\\n" "$*" >> "$FETCH_TRIPWIRE"\nexit 97\n')
        git.chmod(0o755)
        self.sources = {}
        for name in DEPS:
            directory = self.root / (name.lower() + '-src')
            directory.mkdir()
            if name == 'GGUFLIB':
                (directory / 'gguflib.c').write_text('/* source-only fixture */\n')
                (directory / 'fp16.c').write_text('/* source-only fixture */\n')
            else:
                (directory / 'CMakeLists.txt').write_text(f'project({name.lower()} NONE)\n')
            self.sources[name] = directory
        declarations = '\n'.join(
            f'FetchContent_Declare({name.lower()} GIT_REPOSITORY https://invalid.invalid/{name.lower()} GIT_TAG fixed-fixture)' for name in DEPS)
        self.project = ('cmake_minimum_required(VERSION 3.25)\nproject(offline_boundary NONE)\n'
                        'include(FetchContent)\n' + declarations + '\n'
                        'FetchContent_MakeAvailable(json fmt nanobind gguflib)\n'
                        'file(WRITE "${CMAKE_BINARY_DIR}/report" '
                        '"${FETCHCONTENT_FULLY_DISCONNECTED};${FETCHCONTENT_UPDATES_DISCONNECTED};'
                        '${FETCHCONTENT_TRY_FIND_PACKAGE_MODE};${MLX_BUILD_CPU};${MLX_BUILD_METAL};${MLX_BUILD_CUDA}\\n'
                        '${json_SOURCE_DIR}\\n${fmt_SOURCE_DIR}\\n${nanobind_SOURCE_DIR}\\n${gguflib_SOURCE_DIR}\\n")\n')
        (self.source / 'CMakeLists.txt').write_text(self.project)
        self.calls = 0

    def configure(self, missing=None):
        self.calls += 1
        build = self.root / f'build-{self.calls}'
        cmake = shutil.which('cmake')
        assert cmake is not None, 'CMake is required for this configuration regression'
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('CMAKE_', 'FETCHCONTENT_'))}
        env.update(PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                   FETCH_TRIPWIRE=str(self.tripwire), CMAKE=cmake,
                   SOURCE=str(self.source), BUILD=str(build),
                   CMAKE_ARGS='-DMLX_BUILD_CPU=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=OFF')
        env.update({f'FETCHCONTENT_SOURCE_DIR_{name}': str(path)
                    for name, path in self.sources.items() if name != missing})
        # Before this configuration exists, exercise the actual unbound old boundary.
        if CONFIG.exists():
            env['CMAKE_TOOLCHAIN_FILE'] = str(CONFIG)
        assignment = next(line for line in (REPO / 'scripts/build-wheel.sh').read_text().splitlines()
                          if line.startswith('export CMAKE_ARGS="'))
        command = assignment + '\nexec "$CMAKE" -S "$SOURCE" -B "$BUILD" $CMAKE_ARGS\n'
        result = subprocess.run(['bash', '-c', command], env=env,
                                capture_output=True, text=True, timeout=30, check=False)
        return result, build

    def assert_no_download(self):
        self.assertFalse(self.tripwire.exists(), self.tripwire.read_text() if self.tripwire.exists() else '')

    def test_cached_sources_propagate_without_fetch_or_runtime_override(self):
        result, build = self.configure()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_no_download()
        report = (build / 'report').read_text().splitlines()
        self.assertEqual(report[0], 'ON;ON;NEVER;ON;OFF;OFF')
        self.assertEqual(report[1:], [str(self.sources[name]) for name in DEPS])
        self.assertTrue((build / 'CMakeCache.txt').is_file())

    def test_missing_each_source_refuses_before_fetch(self):
        for name in DEPS:
            with self.subTest(dependency=name):
                result, _ = self.configure(missing=name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f'offline dependency {name}', result.stderr)
                self.assert_no_download()

    def test_absent_source_directory_refuses_before_fetch(self):
        shutil.rmtree(self.sources['NANOBIND'])
        result, _ = self.configure()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('offline dependency NANOBIND', result.stderr)
        self.assert_no_download()

    def test_incomplete_gguflib_refuses_before_fetch(self):
        (self.sources['GGUFLIB'] / 'fp16.c').unlink()
        result, _ = self.configure()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('offline dependency GGUFLIB', result.stderr)
        self.assert_no_download()

    def test_undeclared_nested_source_cannot_download(self):
        (self.sources['NANOBIND'] / 'CMakeLists.txt').write_text(
            'FetchContent_Declare(nested GIT_REPOSITORY https://invalid.invalid/nested GIT_TAG fixture)\n'
            'FetchContent_MakeAvailable(nested)\n'
            'if(NOT EXISTS "${nested_SOURCE_DIR}/CMakeLists.txt")\n'
            '  message(FATAL_ERROR "missing nested source")\nendif()\n')
        result, _ = self.configure()
        self.assertNotEqual(result.returncode, 0)
        self.assert_no_download()

    def test_url_source_binding_does_not_attempt_download(self):
        (self.source / 'CMakeLists.txt').write_text(self.project.replace(
            'GIT_REPOSITORY https://invalid.invalid/json GIT_TAG fixed-fixture',
            'URL "file://${CMAKE_SOURCE_DIR}/never-present-json.tar.xz"'))
        result, _ = self.configure()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_no_download()


if __name__ == '__main__':
    unittest.main()
