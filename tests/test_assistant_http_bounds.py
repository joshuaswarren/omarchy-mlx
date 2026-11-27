"""Static file reads and request framing in the assistant HTTP server.

Every input is finite: sparse files stay a few MiB, and each FIFO open runs
under a SIGALRM watchdog so a regression fails instead of hanging the run."""

import contextlib
import json
import os
import signal
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from mlx_omarchy_assistant import server as server_module
from mlx_omarchy_assistant.server import STATIC_MAX_BYTES, AssistantServer, _read_static


class Hung(BaseException):
    """Raised by the watchdog; BaseException so no except Exception swallows it."""


@contextlib.contextmanager
def watchdog(seconds=5):
    def fire(signum, frame):
        raise Hung("blocked for %s s" % seconds)
    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sparse(path, size):
    with open(path, "wb") as stream:
        stream.truncate(size)


class ReadStaticTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_regular_file_up_to_the_cap_is_read_whole(self):
        page = self.root / "page.js"
        page.write_bytes(b"console.log(1)\n")
        self.assertEqual(bytes(_read_static(page)), b"console.log(1)\n")
        full = self.root / "full.bin"
        sparse(full, STATIC_MAX_BYTES)
        self.assertEqual(len(_read_static(full)), STATIC_MAX_BYTES)

    def test_short_reads_are_not_end_of_file_and_never_pass_the_cap(self):
        full = self.root / "full.bin"
        sparse(full, STATIC_MAX_BYTES)
        big = self.root / "big.bin"
        sparse(big, 2 * STATIC_MAX_BYTES)
        real_read = os.read
        total = []

        def short_read(fd, size):
            chunk = real_read(fd, min(size, 65536))
            total.append(len(chunk))
            return chunk
        with mock.patch.object(server_module.os, "read", short_read):
            self.assertEqual(len(_read_static(full)), STATIC_MAX_BYTES)
            total.clear()
            self.assertIsNone(_read_static(big))
        self.assertEqual(sum(total), STATIC_MAX_BYTES + 1)

    def test_fifo_is_refused_without_blocking(self):
        fifo = self.root / "pipe.js"
        os.mkfifo(fifo)
        with watchdog():
            self.assertIsNone(_read_static(fifo))

    def test_directory_missing_file_and_symlink_are_refused(self):
        (self.root / "dir").mkdir()
        page = self.root / "page.js"
        page.write_bytes(b"x")
        (self.root / "link.js").symlink_to(page)
        self.assertIsNone(_read_static(self.root / "dir"))
        self.assertIsNone(_read_static(self.root / "absent.js"))
        self.assertIsNone(_read_static(self.root / "link.js"))


class StaticRouteTests(unittest.TestCase):
    """The route resolves symlinks first, so links that stay inside static/
    still serve and links that leave it are 404 before any open."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.static = self.root / "pkg" / "static"
        self.static.mkdir(parents=True)
        patcher = mock.patch.object(server_module, "__file__", str(self.root / "pkg" / "server.py"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.server = AssistantServer(("127.0.0.1", 0), self.root / "home")
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join(2)
        self.addCleanup(stop)

    def get(self, path):
        return raw_request(self.server.server_port, "GET %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n"
                           % (path, self.server.server_port))

    def test_static_targets(self):
        (self.static / "index.html").write_bytes(b"<!doctype html>")
        (self.static / "alias.html").symlink_to(self.static / "index.html")
        (self.root / "secret.txt").write_bytes(b"secret")
        (self.static / "escape.txt").symlink_to(self.root / "secret.txt")
        (self.static / "js").mkdir()
        sparse(self.static / "huge.js", STATIC_MAX_BYTES + 1)
        status, body = self.get("/")
        self.assertEqual((status, body.split(b"\r\n\r\n", 1)[1]), (200, b"<!doctype html>"))
        self.assertEqual(self.get("/alias.html")[0], 200)
        for path in ("/escape.txt", "/js", "/huge.js", "/absent.js"):
            status, body = self.get(path)
            self.assertEqual(status, 404, path)
            self.assertNotIn(b"secret", body)


def raw_request(port, head, body=b""):
    """Send exact request bytes; returns (status, whole response)."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
        conn.sendall(head.encode("latin-1") + body)
        conn.shutdown(socket.SHUT_WR)
        response = b""
        while len(response) < 1 << 20:
            chunk = conn.recv(65536)
            if not chunk:
                break
            response += chunk
    return int(response.split(b" ", 2)[1]), response


class ContentLengthTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.server = AssistantServer(("127.0.0.1", 0), Path(temp.name))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join(2)
        self.addCleanup(stop)
        self.port = self.server.server_port

    def post(self, *extra):
        """POST {} to the session route with the given extra header lines."""
        head = ("POST /api/session HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nOrigin: http://127.0.0.1:%d\r\n"
                "Content-Type: application/json\r\n" % (self.port, self.port))
        status, response = raw_request(self.port, head + "".join(line + "\r\n" for line in extra) + "\r\n", b"{}")
        return status, json.loads(response.split(b"\r\n\r\n", 1)[1])

    # A body that framed correctly reaches the route, which refuses the
    # missing token with this 401.
    ROUTED = (401, {"error": "This launch link expired. Reopen MLX Chat"})
    REFUSED = (400, {"error": "Send one decimal Content-Length"})

    def test_plain_decimal_length_is_accepted(self):
        self.assertEqual(self.post("Content-Length: 2"), self.ROUTED)
        self.assertEqual(self.post("Content-Length: 002"), self.ROUTED)
        self.assertEqual(self.post("Content-Length: 2 \t"), self.ROUTED)

    def test_missing_length_is_an_empty_body(self):
        self.assertEqual(self.post(), (400, {"error": "Expecting value: line 1 column 1 (char 0)"}))

    def test_non_decimal_lengths_are_refused(self):
        for value in ("+2", "1_1", "2.0", "0x2", "2,2", "\x0b2", "2\x0c", "", "-0"):
            self.assertEqual(self.post("Content-Length: " + value), self.REFUSED, repr(value))

    def test_repeated_lengths_are_refused(self):
        self.assertEqual(self.post("Content-Length: 2", "Content-Length: 2"), self.REFUSED)
        self.assertEqual(self.post("Content-Length: 2", "Content-Length: 900"), self.REFUSED)
        self.assertEqual(self.post("Content-Length: 2", "content-length: 2"), self.REFUSED)

    def test_any_transfer_encoding_header_is_refused(self):
        chunked = (400, {"error": "Chunked uploads are not supported"})
        self.assertEqual(self.post("Content-Length: 2", "Transfer-Encoding: chunked"), chunked)
        self.assertEqual(self.post("Content-Length: 2", "Transfer-Encoding:", "Transfer-Encoding: chunked"),
                         chunked)


if __name__ == "__main__":
    unittest.main()
