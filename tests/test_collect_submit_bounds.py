#!/usr/bin/env python3
"""Focused bounded-read, truncation, and redirect-refusal tests for
scripts/collect_submit.py.

Every test uses finite inputs and loopback-only sockets so the suite runs
cleanly under strict address-space limits (`prlimit --as=2147483648`) on
both Python 3.11 and Python 3.14 with the standard library only.
"""

import array
import http.client
import http.server
import inspect
import os
import socket
import sys
import threading
import unittest
import urllib.error
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import collect_submit as cs  # noqa: E402

PAYLOAD_DEEP = {
    "schema_version": 1,
    "kind": "deep",
    "generated_at": "2026-11-27T00:00:00Z",
}
PAYLOAD_QUICK = {
    "schema_version": 1,
    "kind": "quick",
    "generated_at": "2026-11-27T00:00:00Z",
}


def _no_proxy_env():
    """Patch context clearing proxy env vars so loopback calls stay direct."""
    clean = {k: v for k, v in os.environ.items()
             if not k.lower().endswith("_proxy")}
    return patch.dict(os.environ, clean, clear=True)


def _exact_json_body(target_bytes):
    """Valid duplicate-receipt JSON payload of exactly `target_bytes` bytes."""
    prefix = b'{"status":"duplicate","receipt_url":"http://r/ok","pad":"'
    suffix = b'"}'
    pad_len = target_bytes - len(prefix) - len(suffix)
    assert pad_len >= 0
    body = prefix + (b"a" * pad_len) + suffix
    assert len(body) == target_bytes
    return body


class _SpyStream:
    """Finite streaming response tracking requested chunk sizes, emitted bytes,
    and close state."""

    def __init__(self, status=200, data=b"", headers=None, *,
                 max_chunk=None, length=None, close_error=None,
                 total_bytes=None, fill_byte=b"A"):
        self.status = status
        self.headers = {} if headers is None else headers
        self._data = data
        self._total = len(data) if total_bytes is None else total_bytes
        self._fill_byte = fill_byte
        self._pos = 0
        self._max_chunk = max_chunk
        self._close_error = close_error
        if length is not None:
            self.length = length
        self.read_calls = []
        self.bytes_emitted = 0
        self.closed = False

    def read(self, amt=None):
        self.read_calls.append(amt)
        if self.closed or self._pos >= self._total:
            return b""
        remaining = self._total - self._pos
        if amt is None or amt < 0:
            take = remaining
        else:
            take = min(amt, remaining)
            if self._max_chunk is not None:
                take = min(take, self._max_chunk)
        if self._data:
            out = self._data[self._pos:self._pos + take]
        else:
            out = self._fill_byte * take
        self._pos += len(out)
        self.bytes_emitted += len(out)
        return out

    def close(self):
        self.closed = True
        if self._close_error is not None:
            raise self._close_error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def _start_http_server(testcase, handler_cls):
    """Start a loopback HTTPServer and register deterministic shutdown + join."""
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def stop():
        server.shutdown()
        thread.join(5.0)
        server.server_close()
        testcase.assertFalse(thread.is_alive(), "HTTP server thread did not exit")

    testcase.addCleanup(stop)
    return server


def _start_raw_tcp_server(testcase, raw_response, *, hold_open=False):
    """Serve one raw HTTP response over loopback TCP; return port."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    release = threading.Event()

    def serve():
        conn = None
        try:
            srv.settimeout(2.0)
            conn, _ = srv.accept()
            conn.settimeout(2.0)
            try:
                conn.recv(4096)
            except OSError:
                pass
            conn.sendall(raw_response)
            if hold_open:
                release.wait(2.0)
        except OSError:
            pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
            try:
                srv.close()
            except OSError:
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    def stop():
        release.set()
        try:
            srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            srv.close()
        except OSError:
            pass
        thread.join(5.0)
        testcase.assertFalse(thread.is_alive(), "raw TCP server thread did not exit")

    testcase.addCleanup(stop)
    return port


class ContentLengthPrecheckTests(unittest.TestCase):
    """Upfront Content-Length validation before reading any body bytes."""

    def test_response_cap_constant_matches_payload_cap(self):
        self.assertEqual(cs.MAX_RESPONSE_BYTES, cs.MAX_PAYLOAD_BYTES)
        self.assertEqual(cs.MAX_RESPONSE_BYTES, 256 * 1024)

    def test_oversized_content_length_rejected_before_any_read(self):
        over = str(cs.MAX_RESPONSE_BYTES + 1)
        for case in ("200", "500"):
            with self.subTest(case=case):
                stream = _SpyStream(
                    status=int(case),
                    headers={"Content-Length": over},
                    total_bytes=cs.MAX_RESPONSE_BYTES + 1,
                )
                if case == "200":
                    opener = lambda req, timeout=None, s=stream: s
                else:
                    err = urllib.error.HTTPError(
                        "http://endpoint.example/v1/submit/x", 500, "Err",
                        stream.headers, stream)
                    opener = lambda req, timeout=None, e=err: (_ for _ in ()).throw(e)
                with self.assertRaises(cs.SubmitError) as ctx:
                    cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                              urlopen=opener)
                self.assertIn("Content-Length", str(ctx.exception))
                self.assertIn(str(cs.MAX_RESPONSE_BYTES), str(ctx.exception))
                self.assertEqual(stream.read_calls, [])
                self.assertEqual(stream.bytes_emitted, 0)
                self.assertTrue(stream.closed)

    def test_exact_limit_content_length_succeeds(self):
        body = _exact_json_body(cs.MAX_RESPONSE_BYTES)
        stream = _SpyStream(
            status=200,
            data=body,
            headers={"Content-Length": str(cs.MAX_RESPONSE_BYTES)},
            max_chunk=16384,
        )
        res = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                        urlopen=lambda req, timeout=None: stream)
        self.assertEqual(res["url"], "http://r/ok")
        self.assertTrue(res["deduplicated"])
        self.assertEqual(stream.bytes_emitted, cs.MAX_RESPONSE_BYTES)
        self.assertTrue(stream.closed)

    def test_leading_zeros_at_and_above_limit(self):
        body = _exact_json_body(cs.MAX_RESPONSE_BYTES)
        stream_ok = _SpyStream(
            status=200,
            data=body,
            headers={"Content-Length": ("0" * 50) + str(cs.MAX_RESPONSE_BYTES)},
        )
        res = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                        urlopen=lambda req, timeout=None: stream_ok)
        self.assertEqual(res["url"], "http://r/ok")

        stream_zero = _SpyStream(
            status=200,
            data=b'{"status":"duplicate","receipt_url":"http://r/z"}',
            headers={"Content-Length": "0" * 5000},
        )
        res_z = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                          urlopen=lambda req, timeout=None: stream_zero)
        self.assertEqual(res_z["url"], "http://r/z")

        stream_over = _SpyStream(
            status=200,
            headers={"Content-Length": ("0" * 50) + str(cs.MAX_RESPONSE_BYTES + 1)},
            total_bytes=10,
        )
        with self.assertRaises(cs.SubmitError) as ctx:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: stream_over)
        self.assertIn(str(cs.MAX_RESPONSE_BYTES + 1), str(ctx.exception))
        self.assertEqual(stream_over.read_calls, [])

    def test_huge_digit_count_content_length_avoids_int_str_limit(self):
        # 10,000 digits exceeds Python 3.11+'s default 4,300-digit int() limit.
        huge = "9" * 10000
        stream = _SpyStream(
            status=200,
            headers={"Content-Length": huge},
            total_bytes=10,
        )
        with self.assertRaises(cs.SubmitError) as ctx:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: stream)
        msg = str(ctx.exception)
        self.assertIn("too large", msg)
        self.assertIn("9" * 32 + "...", msg)
        self.assertNotIn("endpoint unreachable", msg)
        self.assertEqual(stream.read_calls, [])

    def test_ows_stripped_and_non_decimal_headers_fall_through_to_stream_cap(self):
        over = str(cs.MAX_RESPONSE_BYTES + 1)
        stream_ows = _SpyStream(
            status=200,
            headers={"Content-Length": f" \t{over}\t "},
            total_bytes=10,
        )
        with self.assertRaises(cs.SubmitError):
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: stream_ows)
        self.assertEqual(stream_ows.read_calls, [])

        ok_body = b'{"status":"duplicate","receipt_url":"http://r/ok"}'
        non_strict = (
            f"+{over}",
            "-1",
            "1_000_000",
            "abc",
            "\u0669\u0669\u0669\u0669\u0669\u0669",  # Arabic-Indic digits
            f"\n{over}",
            f"{over}\n",
            None,
            9999999,
        )
        for bad in non_strict:
            with self.subTest(bad=bad, mode="small"):
                s = _SpyStream(status=200, data=ok_body,
                               headers={"Content-Length": bad})
                res = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                                urlopen=lambda req, timeout=None, st=s: st)
                self.assertEqual(res["url"], "http://r/ok")
            with self.subTest(bad=bad, mode="oversized_stream"):
                s = _SpyStream(status=200, headers={"Content-Length": bad},
                               total_bytes=cs.MAX_RESPONSE_BYTES + 4096)
                with self.assertRaises(cs.SubmitError) as ctx:
                    cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                              urlopen=lambda req, timeout=None, st=s: st)
                self.assertIn("too large", str(ctx.exception))
                self.assertEqual(s.bytes_emitted, cs.MAX_RESPONSE_BYTES + 1)

    def test_multiple_comma_separated_and_case_insensitive_content_length(self):
        over = str(cs.MAX_RESPONSE_BYTES + 1)

        msg = http.client.HTTPMessage()
        msg.add_header("Content-Length", "10")
        msg.add_header("Content-Length", over)
        s_multi = _SpyStream(status=200, headers=msg, total_bytes=10)
        with self.assertRaises(cs.SubmitError):
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: s_multi)
        self.assertEqual(s_multi.read_calls, [])

        s_comma = _SpyStream(
            status=200,
            headers={"content-length": f"10, {over}"},
            total_bytes=10,
        )
        with self.assertRaises(cs.SubmitError):
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: s_comma)
        self.assertEqual(s_comma.read_calls, [])

    def test_header_helpers_none_case_insensitive_and_get_only_stub(self):
        class GetOnlyHeaders:
            def __init__(self, mapping):
                self._m = mapping

            def get(self, key, default=None):
                return self._m.get(key, default)

        # headers=None on normal response
        s_none = _SpyStream(status=200,
                            data=b'{"status":"duplicate","receipt_url":"http://r/n"}')
        s_none.headers = None
        status, _, ray = cs._request(
            lambda req, timeout=None: s_none, None, 5)
        self.assertEqual(status, 200)
        self.assertIsNone(ray)

        # Case-insensitive cf-ray in dict with a non-str key present
        s_ray = _SpyStream(status=200, data=b"{}",
                           headers={123: "ignored", "CF-Ray": "ray-upper"})
        _, _, ray = cs._request(lambda req, timeout=None: s_ray, None, 5)
        self.assertEqual(ray, "ray-upper")

        # Get-only stub works for both cf-ray and Content-Length rejection
        s_get_ok = _SpyStream(status=200, data=b"{}",
                              headers=GetOnlyHeaders({"cf-ray": "ray-get"}))
        _, _, ray = cs._request(lambda req, timeout=None: s_get_ok, None, 5)
        self.assertEqual(ray, "ray-get")

        s_get_big = _SpyStream(
            status=200,
            headers=GetOnlyHeaders({"Content-Length": str(cs.MAX_RESPONSE_BYTES + 1)}),
            total_bytes=10,
        )
        with self.assertRaises(cs.SubmitError):
            cs._request(lambda req, timeout=None: s_get_big, None, 5)
        self.assertEqual(s_get_big.read_calls, [])

        # Object with no header methods at all
        self.assertEqual(cs._header_values(object(), "cf-ray"), [])


class BoundedStreamReadTests(unittest.TestCase):
    """Chunked bounded reading, OOM defense, cleanup safety, and stub compat."""

    def test_oversized_200_stream_stops_at_cap_plus_one_and_uses_bounded_chunks(self):
        # 8 MiB finite virtual stream (32x cap): stops at MAX_RESPONSE_BYTES + 1.
        stream = _SpyStream(status=200, total_bytes=8 * 1024 * 1024)
        with self.assertRaises(cs.SubmitError) as ctx:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: stream)
        self.assertIn("too large", str(ctx.exception))
        self.assertEqual(stream.bytes_emitted, cs.MAX_RESPONSE_BYTES + 1)
        self.assertTrue(stream.closed)
        self.assertTrue(all(
            isinstance(amt, int) and 0 < amt <= cs._READ_CHUNK_BYTES
            for amt in stream.read_calls))
        # Final request asks for exactly 1 byte past MAX_RESPONSE_BYTES.
        self.assertEqual(stream.read_calls[-1], 1)

    def test_varargs_posonly_and_unsignaturable_readers_use_bounded_chunks(self):
        class VarArgsStream(_SpyStream):
            def read(self, *args):
                amt = args[0] if args else None
                return super().read(amt)

        class PosOnlyStream(_SpyStream):
            def read(self, amt=None, /):
                return super().read(amt)

        for cls in (VarArgsStream, PosOnlyStream):
            with self.subTest(cls=cls.__name__):
                s = cls(status=200, total_bytes=8 * 1024 * 1024)
                with self.assertRaises(cs.SubmitError) as ctx:
                    cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                              urlopen=lambda req, timeout=None, st=s: st)
                self.assertIn("too large", str(ctx.exception))
                self.assertEqual(s.bytes_emitted, cs.MAX_RESPONSE_BYTES + 1)
                self.assertTrue(all(
                    isinstance(amt, int) and 0 < amt <= cs._READ_CHUNK_BYTES
                    for amt in s.read_calls))

        s_unsig = _SpyStream(status=200, total_bytes=8 * 1024 * 1024)
        with patch.object(inspect, "signature",
                          side_effect=ValueError("no signature")):
            with self.assertRaises(cs.SubmitError) as ctx2:
                cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                          urlopen=lambda req, timeout=None: s_unsig)
        self.assertIn("too large", str(ctx2.exception))
        self.assertEqual(s_unsig.bytes_emitted, cs.MAX_RESPONSE_BYTES + 1)
        self.assertTrue(all(
            isinstance(amt, int) and 0 < amt <= cs._READ_CHUNK_BYTES
            for amt in s_unsig.read_calls))

    def test_oversized_httperror_stream_bounded_and_closed(self):
        fp = _SpyStream(status=500, total_bytes=8 * 1024 * 1024)
        err = urllib.error.HTTPError(
            "http://endpoint.example/v1/submit/x", 500, "Server Error",
            {"cf-ray": "ray-500"}, fp)
        with self.assertRaises(cs.SubmitError) as ctx:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: (_ for _ in ()).throw(err))
        self.assertIn("too large", str(ctx.exception))
        self.assertEqual(fp.bytes_emitted, cs.MAX_RESPONSE_BYTES + 1)
        self.assertTrue(fp.closed)

    def test_normal_httperror_closes_response_and_preserves_detail(self):
        fp = _SpyStream(status=422, data=b'{"error":"schema_invalid"}')
        err = urllib.error.HTTPError(
            "http://endpoint.example/v1/submit/x", 422, "Unprocessable",
            {"cf-ray": "ray-422"}, fp)
        with self.assertRaises(cs.SubmitError) as ctx:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: (_ for _ in ()).throw(err))
        msg = str(ctx.exception)
        self.assertIn("HTTP 422", msg)
        self.assertIn("schema_invalid", msg)
        self.assertIn("ray-422", msg)
        self.assertTrue(fp.closed)

    def test_close_errors_never_mask_submit_error_on_2xx_or_httperror(self):
        for exc_cls in (OSError, RuntimeError):
            with self.subTest(path="2xx", exc=exc_cls.__name__):
                s = _SpyStream(
                    status=200,
                    total_bytes=cs.MAX_RESPONSE_BYTES + 4096,
                    close_error=exc_cls("close boom"),
                )
                with self.assertRaises(cs.SubmitError) as ctx:
                    cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                              urlopen=lambda req, timeout=None, st=s: st)
                self.assertIn("too large", str(ctx.exception))
                self.assertNotIn("endpoint unreachable", str(ctx.exception))
                self.assertTrue(s.closed)

            with self.subTest(path="HTTPError", exc=exc_cls.__name__):
                fp = _SpyStream(
                    status=500,
                    total_bytes=cs.MAX_RESPONSE_BYTES + 4096,
                    close_error=exc_cls("close boom"),
                )
                err = urllib.error.HTTPError(
                    "http://endpoint.example/v1/submit/x", 500, "Err", {}, fp)
                with self.assertRaises(cs.SubmitError) as ctx:
                    cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                              urlopen=lambda req, timeout=None, e=err: (_ for _ in ()).throw(e))
                self.assertIn("too large", str(ctx.exception))
                self.assertNotIn("endpoint unreachable", str(ctx.exception))
                self.assertTrue(fp.closed)

        # Non-context-manager response with only close() is still closed.
        class CloseOnly:
            status = 200
            headers = {}

            def __init__(self):
                self._chunks = [
                    b'{"status":"duplicate","receipt_url":"http://r/co"}'
                ]
                self.closed = False

            def read(self, amt=None):
                return self._chunks.pop(0) if self._chunks else b""

            def close(self):
                self.closed = True

        co = CloseOnly()
        res = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                        urlopen=lambda req, timeout=None: co)
        self.assertEqual(res["url"], "http://r/co")
        self.assertTrue(co.closed)

        # Context-manager response with __enter__/__exit__ only (no close())
        # must have __exit__ invoked (including if __enter__ raises).
        class ExitOnlyCM:
            status = 200
            headers = {}

            def __init__(self, enter_error=None):
                self._chunks = [
                    b'{"status":"duplicate","receipt_url":"http://r/cm"}'
                ]
                self._enter_error = enter_error
                self.exited = False

            def read(self, amt=None):
                return self._chunks.pop(0) if self._chunks else b""

            def __enter__(self):
                if self._enter_error is not None:
                    raise self._enter_error
                return self

            def __exit__(self, *args):
                self.exited = True
                return False

        cm_ok = ExitOnlyCM()
        res_cm = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                           urlopen=lambda req, timeout=None: cm_ok)
        self.assertEqual(res_cm["url"], "http://r/cm")
        self.assertTrue(cm_ok.exited)

        cm_fail = ExitOnlyCM(enter_error=OSError("enter boom"))
        with self.assertRaises(cs.SubmitError):
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: cm_fail)
        self.assertTrue(cm_fail.exited)

    def test_lying_small_content_length_still_capped_by_stream(self):
        stream = _SpyStream(
            status=200,
            headers={"Content-Length": "10"},
            total_bytes=cs.MAX_RESPONSE_BYTES + 1000,
        )
        with self.assertRaises(cs.SubmitError) as ctx:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: stream)
        self.assertIn("too large", str(ctx.exception))
        self.assertEqual(stream.bytes_emitted, cs.MAX_RESPONSE_BYTES + 1)

    def test_single_oversized_chunk_from_noncompliant_reader_rejected(self):
        class GreedyReader:
            status = 200
            headers = {}

            def read(self, amt=None):
                return b"x" * (cs.MAX_RESPONSE_BYTES + 1)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with self.assertRaises(cs.SubmitError) as ctx:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: GreedyReader())
        self.assertIn("too large", str(ctx.exception))

    def test_zero_arg_and_kwonly_fake_responses_called_once_and_capped(self):
        class ZeroArgResp:
            status = 200
            headers = {}

            def __init__(self, body):
                self._body = body
                self.calls = 0

            def read(self):
                self.calls += 1
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class KwOnlyResp(ZeroArgResp):
            def read(self, *, amt=None):
                self.calls += 1
                return self._body

        ok_body = _exact_json_body(cs.MAX_RESPONSE_BYTES)
        for cls in (ZeroArgResp, KwOnlyResp):
            with self.subTest(cls=cls.__name__, mode="ok"):
                r = cls(ok_body)
                res = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                                urlopen=lambda req, timeout=None, resp=r: resp)
                self.assertEqual(res["url"], "http://r/ok")
                self.assertEqual(r.calls, 1)

            with self.subTest(cls=cls.__name__, mode="over"):
                r = cls(b"x" * (cs.MAX_RESPONSE_BYTES + 1))
                with self.assertRaises(cs.SubmitError) as ctx:
                    cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                              urlopen=lambda req, timeout=None, resp=r: resp)
                self.assertIn("too large", str(ctx.exception))
                self.assertEqual(r.calls, 1)

        # Zero-arg memoryview must convert to bytes so _decode() succeeds.
        raw_mv = memoryview(b'{"status":"duplicate","receipt_url":"http://r/zmv"}')
        r_zmv = ZeroArgResp(raw_mv)
        res_zmv = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                            urlopen=lambda req, timeout=None: r_zmv)
        self.assertEqual(res_zmv["url"], "http://r/zmv")
        self.assertEqual(r_zmv.calls, 1)

        # Non-byte-format memoryview uses nbytes (not element len) for cap check.
        # 70,000 uint32s = 280,000 bytes > 262,144, though len(mv) == 70,000 <= 262,144.
        wide_mv = memoryview(array.array("I", [0]) * 70000)
        self.assertLessEqual(len(wide_mv), cs.MAX_RESPONSE_BYTES)
        self.assertGreater(wide_mv.nbytes, cs.MAX_RESPONSE_BYTES)
        r_wide = ZeroArgResp(wide_mv)
        with self.assertRaises(cs.SubmitError) as ctx_w:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: r_wide)
        self.assertIn("too large", str(ctx_w.exception))

    def test_bytearray_memoryview_accepted_and_non_bytes_including_falsy_rejected(self):
        class TypedReader:
            status = 200
            headers = {}

            def __init__(self, items, zero_arg=False):
                self._items = list(items)
                if zero_arg:
                    self.read = self._read0
                else:
                    self.read = self._read1

            def _read0(self):
                return self._items.pop(0)

            def _read1(self, amt=None):
                return self._items.pop(0) if self._items else b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        raw = b'{"status":"duplicate","receipt_url":"http://r/mv"}'
        # Include a strided non-contiguous memoryview to match zero-arg bytes().
        interleaved = bytearray()
        for byte in raw[10:]:
            interleaved.append(byte)
            interleaved.append(0)
        strided_mv = memoryview(interleaved)[::2]
        self.assertFalse(strided_mv.c_contiguous)
        r_mv = TypedReader([bytearray(raw[:10]), strided_mv])
        res = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                        urlopen=lambda req, timeout=None: r_mv)
        self.assertEqual(res["url"], "http://r/mv")

        # Sized wide memoryview exceeding cap via nbytes > limit is rejected.
        wide_mv = memoryview(array.array("I", [0]) * 70000)
        r_wide_sized = TypedReader([wide_mv])
        with self.assertRaises(cs.SubmitError) as ctx_ws:
            cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                      urlopen=lambda req, timeout=None: r_wide_sized)
        self.assertIn("too large", str(ctx_ws.exception))

        # Truthy AND falsy non-bytes values must all fail with non-bytes error.
        for bad_val in ("not-bytes", "", None, 0, False, []):
            for zero_arg in (False, True):
                with self.subTest(bad_val=bad_val, zero_arg=zero_arg):
                    r_bad = TypedReader([bad_val], zero_arg=zero_arg)
                    with self.assertRaises(cs.SubmitError) as ctx:
                        cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                                  urlopen=lambda req, timeout=None, r=r_bad: r)
                    self.assertIn("non-bytes", str(ctx.exception))

    def test_oversized_chunked_and_combined_headers_over_real_loopback(self):
        chunk_hex = f"{65536:x}\r\n".encode("ascii")
        chunk_body = b"Z" * 65536 + b"\r\n"
        body_1m = (chunk_hex + chunk_body) * 16 + b"0\r\n\r\n"

        for label, raw in (
                ("chunked_200",
                 b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + body_1m),
                ("chunked_500",
                 b"HTTP/1.1 500 Err\r\nTransfer-Encoding: chunked\r\n\r\n" + body_1m),
                ("chunked_with_small_cl",
                 b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
                 b"Content-Length: 10\r\n\r\n" + body_1m),
                ("chunked_with_oversized_cl",
                 b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
                 + f"Content-Length: {cs.MAX_RESPONSE_BYTES + 1}\r\n\r\n".encode("ascii")
                 + b"2\r\n{}\r\n0\r\n\r\n"),
        ):
            with self.subTest(label=label):
                port = _start_raw_tcp_server(self, raw)
                with _no_proxy_env():
                    with self.assertRaises(cs.SubmitError) as ctx:
                        cs.submit(f"http://127.0.0.1:{port}", b"x",
                                  PAYLOAD_DEEP, timeout=5)
                self.assertIn("too large", str(ctx.exception))


class TruncatedFramedReadTests(unittest.TestCase):
    """Short Content-Length and chunked responses must surface as SubmitError."""

    def test_short_framed_200_loopback_raises_submit_error(self):
        partial = b'{"status":"duplicate","receipt_url":"http://r/1"}'
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 200\r\n\r\n" + partial
        )
        for name, call in (
                ("submit", lambda url: cs.submit(url, b"x", PAYLOAD_DEEP,
                                                 timeout=5)),
                ("submit_payload", lambda url: cs.submit_payload(
                    url, PAYLOAD_QUICK, timeout=5))):
            with self.subTest(name=name):
                port = _start_raw_tcp_server(self, raw)
                with _no_proxy_env():
                    with self.assertRaises(cs.SubmitError) as ctx:
                        call(f"http://127.0.0.1:{port}")
                msg = str(ctx.exception)
                self.assertIn("truncated", msg)
                self.assertIn(str(200 - len(partial)), msg)

    def test_short_framed_500_httperror_loopback_raises_submit_error(self):
        partial = b'{"error":"boom"}'
        raw = (
            b"HTTP/1.1 500 Internal Server Error\r\n"
            b"Content-Length: 200\r\n\r\n" + partial
        )
        port = _start_raw_tcp_server(self, raw)
        with _no_proxy_env():
            with self.assertRaises(cs.SubmitError) as ctx:
                cs.submit(f"http://127.0.0.1:{port}", b"x", PAYLOAD_DEEP,
                          timeout=5)
        msg = str(ctx.exception)
        self.assertIn("truncated", msg)
        self.assertIn(str(200 - len(partial)), msg)

    def test_short_framed_empty_body_loopback_raises_submit_error(self):
        raw = b"HTTP/1.1 200 OK\r\nContent-Length: 50\r\n\r\n"
        port = _start_raw_tcp_server(self, raw)
        with _no_proxy_env():
            with self.assertRaises(cs.SubmitError) as ctx:
                cs.submit(f"http://127.0.0.1:{port}", b"x", PAYLOAD_DEEP,
                          timeout=5)
        self.assertIn("truncated: 50 bytes remaining", str(ctx.exception))

    def test_short_chunked_loopback_raises_submit_error(self):
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhello\r\n"
        )
        port = _start_raw_tcp_server(self, raw)
        with _no_proxy_env():
            with self.assertRaises(cs.SubmitError) as ctx:
                cs.submit(f"http://127.0.0.1:{port}", b"x", PAYLOAD_DEEP,
                          timeout=5)
        self.assertIn("IncompleteRead", str(ctx.exception))

    def test_remaining_framed_bytes_fp_fallback_and_type_guards(self):
        class WrapperWithFpLength:
            status = 200
            headers = {}

            def __init__(self, fp_length, outer_length=None, set_outer=False):
                if set_outer:
                    self.length = outer_length
                self.fp = type("FP", (), {"length": fp_length})()

            def read(self, amt=None):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        # Missing resp.length falls back to resp.fp.length.
        with self.assertRaises(cs.SubmitError) as ctx:
            cs._read_bounded(WrapperWithFpLength(12), {})
        self.assertIn("truncated: 12 bytes remaining", str(ctx.exception))

        # Invalid non-int resp.length ("bad", True) must fall through to fp.length.
        for bad_outer in ("bad", True, object()):
            with self.subTest(bad_outer=bad_outer):
                with self.assertRaises(cs.SubmitError) as ctx2:
                    cs._read_bounded(
                        WrapperWithFpLength(19, outer_length=bad_outer, set_outer=True),
                        {})
                self.assertIn("truncated: 19 bytes remaining", str(ctx2.exception))

        # bool (True is an int subclass), 0, and negative values are not truncation.
        for non_short in (True, 0, None, -5):
            with self.subTest(length=non_short):
                s = _SpyStream(status=200, data=b"{}", length=non_short)
                self.assertEqual(cs._read_bounded(s, {}), b"{}")

    def test_keepalive_framed_200_returns_without_waiting_for_close(self):
        body = b'{"status":"duplicate","receipt_url":"http://r/ka"}'
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Connection: keep-alive\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
        port = _start_raw_tcp_server(self, raw, hold_open=True)
        with _no_proxy_env():
            res = cs.submit(f"http://127.0.0.1:{port}", b"x", PAYLOAD_DEEP,
                            timeout=2)
        self.assertEqual(res["url"], "http://r/ka")


class RedirectRefusalTests(unittest.TestCase):
    """Default opener refuses 3xx redirects unread without forwarding headers."""

    def _start_target_server(self):
        hits = []

        class Target(http.server.BaseHTTPRequestHandler):
            def _record(self):
                hits.append({
                    "method": self.command,
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "aliases": self.headers.get(cs.ALIAS_HEADER),
                })
                body = b'{"status":"duplicate","receipt_url":"http://r/leaked"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = _record
            do_POST = _record

            def log_message(self, *args):
                pass

        server = _start_http_server(self, Target)
        return server.server_address[1], hits

    def test_all_redirect_codes_refused_without_following_or_leaking_headers(self):
        t_port, target_hits = self._start_target_server()
        current_code = [302]

        class Redirector(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(current_code[0])
                self.send_header(
                    "Location",
                    f"http://127.0.0.1:{t_port}/steal?secret=do-not-echo")
                self.send_header("cf-ray", f"ray-{current_code[0]}")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        redir = _start_http_server(self, Redirector)
        url = f"http://127.0.0.1:{redir.server_address[1]}"

        with _no_proxy_env():
            for code in (301, 302, 303, 307, 308):
                current_code[0] = code
                for name, call in (
                        ("submit", lambda: cs.submit(
                            url, b"x", PAYLOAD_DEEP, timeout=5,
                            token="secret-token", aliases=["secret-alias"])),
                        ("submit_payload", lambda: cs.submit_payload(
                            url, PAYLOAD_QUICK, timeout=5,
                            token="secret-token", aliases=["secret-alias"]))):
                    with self.subTest(code=code, entry=name):
                        with self.assertRaises(cs.SubmitError) as ctx:
                            call()
                        msg = str(ctx.exception)
                        self.assertIn(f"HTTP {code}", msg)
                        self.assertIn(f"ray-{code}", msg)
                        self.assertIn("to http://127.0.0.1", msg)
                        self.assertNotIn("steal", msg)
                        self.assertNotIn("do-not-echo", msg)
                        self.assertIn("refusing to follow redirects", msg)
        self.assertEqual(target_hits, [])

    def test_post_initiate_307_refused_without_leaking_auth_or_aliases(self):
        t_port, target_hits = self._start_target_server()

        class Probe404Then307(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(404)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    self.rfile.read(length)
                self.send_response(307)
                self.send_header("Location", f"http://127.0.0.1:{t_port}/steal")
                self.send_header("cf-ray", "ray-post-307")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        redir = _start_http_server(self, Probe404Then307)
        url = f"http://127.0.0.1:{redir.server_address[1]}"

        with _no_proxy_env(), patch.object(cs, "solve_pow", return_value="0"):
            for name, call in (
                    ("submit", lambda: cs.submit(
                        url, b"x", PAYLOAD_DEEP, timeout=5,
                        token="secret-token", aliases=["secret-alias"])),
                    ("submit_payload", lambda: cs.submit_payload(
                        url, PAYLOAD_QUICK, timeout=5,
                        token="secret-token", aliases=["secret-alias"]))):
                with self.subTest(entry=name):
                    with self.assertRaises(cs.SubmitError) as ctx:
                        call()
                    msg = str(ctx.exception)
                    self.assertIn("HTTP 307", msg)
                    self.assertIn("ray-post-307", msg)
        self.assertEqual(target_hits, [])

    def test_redirect_location_hint_sanitizes_malformed_and_non_http_targets(self):
        handler = cs._NoRedirectHandler()
        for location, expected_sub, forbidden_sub in (
                ("https://collector.example:8443/v1?k=v",
                 "redirected to https://collector.example", "k=v"),
                ("http://[::1]:8080/v1",
                 "redirected to http://[::1]", "8080"),
                ("http://[::1", None, "redirected to "),
                ("https:///path", None, "redirected to "),
                ("http://", None, "redirected to "),
                ("https://host:bad/v1", None, "redirected to "),
                ("https://host:99999/v1", None, "redirected to "),
                ("https://collector.example/a\nb", None, "redirected to "),
                ("https://collector.example/\tpath", None, "redirected to "),
                ("https://" + ("a" * 256) + "/v1", None, "redirected to "),
                ("ftp://ftp.example/file", None, "redirected to "),
                ("file:///etc/passwd", None, "redirected to "),
                ("https://bad\x1bhost.example/", None, "redirected to "),
                ("https://bad host.example/", None, "redirected to "),
                ("", None, "redirected to "),
        ):
            with self.subTest(location=location):
                fp = _SpyStream(status=302)
                with self.assertRaises(cs.SubmitError) as ctx:
                    handler.http_error_302(
                        None, fp, 302, "Found", {"Location": location})
                msg = str(ctx.exception)
                if expected_sub:
                    self.assertIn(expected_sub, msg)
                if forbidden_sub:
                    self.assertNotIn(forbidden_sub, msg)
                self.assertTrue(fp.closed)

        # Control chars, whitespace, or oversized cf-ray are sanitized to unknown.
        for bad_ray in ("\x1b[2J", "ray\nline", "ray space", "r" * 129):
            with self.subTest(bad_ray=bad_ray[:16]):
                fp = _SpyStream(status=302)
                with self.assertRaises(cs.SubmitError) as ctx:
                    handler.http_error_302(
                        None, fp, 302, "Found", {"cf-ray": bad_ray})
                msg = str(ctx.exception)
                self.assertIn("request unknown", msg)
                self.assertNotIn("\x1b", msg)
                self.assertTrue(fp.closed)

    def test_redirect_body_closed_unread_without_draining(self):
        # Unit check on _NoRedirectHandler directly: zero reads, fp closed.
        handler = cs._NoRedirectHandler()
        fp = _SpyStream(status=302, total_bytes=8 * 1024 * 1024)
        with self.assertRaises(cs.SubmitError) as ctx:
            handler.http_error_302(None, fp, 302, "Found", {"cf-ray": "ray-u"})
        self.assertEqual(fp.read_calls, [])
        self.assertEqual(fp.bytes_emitted, 0)
        self.assertTrue(fp.closed)
        self.assertIn("HTTP 302", str(ctx.exception))
        self.assertIn("ray-u", str(ctx.exception))

        # OSError AND RuntimeError from fp.close() must not replace SubmitError.
        for exc_cls in (OSError, RuntimeError):
            with self.subTest(close_exc=exc_cls.__name__):
                fp_err = _SpyStream(status=302, close_error=exc_cls("bad fd"))
                with self.assertRaises(cs.SubmitError) as ctx2:
                    handler.http_error_302(None, fp_err, 302, "Found", None)
                self.assertIn("HTTP 302", str(ctx2.exception))
                self.assertIn("request unknown", str(ctx2.exception))
                self.assertTrue(fp_err.closed)

        # Loopback check with a 64 MiB finite redirect body: client closes
        # unread instead of draining all 64 MiB.
        t_port, target_hits = self._start_target_server()
        total_body = 64 * 1024 * 1024
        bytes_written = [0]
        handler_done = threading.Event()

        class BigRedirect(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{t_port}/steal")
                    self.send_header("Content-Length", str(total_body))
                    self.end_headers()
                    chunk = b"R" * 65536
                    for _ in range(total_body // len(chunk)):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        bytes_written[0] += len(chunk)
                except OSError:
                    pass
                finally:
                    handler_done.set()

            def log_message(self, *args):
                pass

        redir = _start_http_server(self, BigRedirect)

        with _no_proxy_env():
            with self.assertRaises(cs.SubmitError) as ctx3:
                cs.submit(f"http://127.0.0.1:{redir.server_address[1]}",
                          b"x", PAYLOAD_DEEP, timeout=5)
        self.assertIn("HTTP 302", str(ctx3.exception))
        self.assertTrue(handler_done.wait(5.0), "BigRedirect handler did not finish")
        self.assertEqual(target_hits, [])
        self.assertLess(bytes_written[0], total_body)

    def test_custom_urlopen_injection_preserved(self):
        calls = []

        def custom_open(req, timeout=None):
            calls.append((req.get_full_url(), timeout))
            return _SpyStream(
                status=200,
                data=b'{"status":"duplicate","receipt_url":"http://r/custom"}',
            )

        res1 = cs.submit("http://endpoint.example", b"x", PAYLOAD_DEEP,
                         timeout=12, urlopen=custom_open)
        res2 = cs.submit_payload("http://endpoint.example", PAYLOAD_QUICK,
                                 timeout=13, urlopen=custom_open)
        self.assertEqual(res1["url"], "http://r/custom")
        self.assertEqual(res2["url"], "http://r/custom")
        self.assertEqual([t for _, t in calls], [12, 13])


if __name__ == "__main__":
    unittest.main()
