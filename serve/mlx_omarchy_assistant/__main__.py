"""Launch the local application or attach a terminal to the same coordinator."""

from __future__ import annotations

import argparse
import errno
import fcntl
import functools
import http.client
import json
import os
import secrets
import signal
import stat
import sys
import threading
import time
import webbrowser
from pathlib import Path

import mlx_omarchy_paths

from .server import AssistantServer
from . import wake_word


def request(runtime, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", runtime["port"], timeout=20)
    headers = {"Cookie": runtime["cookie"], "X-Assistant-CSRF": runtime["csrf"],
               "Origin": f"http://127.0.0.1:{runtime['port']}", "Content-Type": "application/json"}
    try:
        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        data = json.loads(response.read(2 * 1024 * 1024))
        if response.status >= 400:
            raise RuntimeError(data.get("error", "Local application request failed"))
        return data
    finally:
        connection.close()


def terminal(runtime, prompt=None, once=False, maximum=None):
    conversation = request(runtime, "POST", "/api/conversations", {"save": False})["id"]
    while True:
        try:
            text = prompt if prompt is not None else input("You: ")
            prompt = None
            if text.strip() in ("/quit", "/exit"):
                return 0
            if not text.strip():
                continue
            payload = {"text": text}
            if maximum is not None:
                payload["max_tokens"] = maximum
            turn = request(runtime, "POST", f"/api/conversations/{conversation}/turns", payload)["turn_id"]
            finished = threading.Event()
            failed = False

            def heartbeat():
                while not finished.wait(3):
                    try:
                        request(runtime, "POST", f"/api/conversations/{conversation}/heartbeat", {"turn_id": turn})
                    except (OSError, RuntimeError):
                        return

            watcher = threading.Thread(target=heartbeat, daemon=True)
            watcher.start()
            cursor = 0
            print("Assistant: ", end="", flush=True)
            try:
                while not finished.is_set():
                    connection = http.client.HTTPConnection("127.0.0.1", runtime["port"], timeout=20)
                    try:
                        connection.request("GET", f"/api/conversations/{conversation}/events?after={cursor}",
                                           headers={"Cookie": runtime["cookie"]})
                        response = connection.getresponse()
                        if response.status != 200:
                            raise RuntimeError("Event stream interrupted; reopen the conversation in MLX Chat")
                        for line in response:
                            if not line.startswith(b"data:"):
                                continue
                            event = json.loads(line[5:])
                            cursor = event["sequence"]
                            if event["turn_id"] != turn:
                                continue
                            if event["type"] == "text":
                                print(event["data"]["text"], end="", flush=True)
                            elif event["type"] == "component":
                                from .components import plain_text
                                print("\n" + plain_text(event["data"]), flush=True)
                            elif event["type"] == "decision":
                                from .coordinator import format_decision_event
                                rendered = format_decision_event(event["data"])
                                if rendered:
                                    print("\n" + rendered, flush=True)
                            elif event["type"] == "error":
                                failed = True
                                print("\nError: " + event["data"]["message"], file=sys.stderr)
                            elif event["type"] == "done":
                                finished.set()
                    finally:
                        connection.close()
            except KeyboardInterrupt:
                request(runtime, "POST", f"/api/conversations/{conversation}/cancel", {"turn_id": turn})
            finally:
                finished.set()
                watcher.join(timeout=4)
                print()
            if once:
                return 1 if failed else 0
        except (EOFError, KeyboardInterrupt):
            return 0


STOP_BUDGET = 4.0


def _remove_runtime_file(path):
    try:
        if stat.S_ISREG(os.lstat(path).st_mode):
            path.unlink()
    except OSError:
        pass


def stop_server(server, runtime_file, *, serving=True):
    """Stop the server within STOP_BUDGET seconds, whatever stage hangs.

    Past the budget the voice workers are killed and the process leaves at
    once: the stop never outlasts the caller's SIGKILL fallback. A
    runtime_file of None means this process never owned one; serving=False
    skips shutdown(), which blocks forever when serve_forever() never ran.
    """
    failures = []

    def run():
        try:
            if serving:
                server.shutdown()
            server.server_close()
        except BaseException as error:
            failures.append(error)

    stopper = threading.Thread(target=run, daemon=True)
    stopper.start()
    stopper.join(STOP_BUDGET)
    if runtime_file is not None:
        _remove_runtime_file(runtime_file)
    if stopper.is_alive():
        from .synthesis import kill_workers

        kill_workers()
        print(f"MLX Chat: stop exceeded {STOP_BUDGET:g}s; workers killed", file=sys.stderr, flush=True)
        sys.stdout.flush()
        os._exit(0)
    if failures:
        raise failures[0]


RUNTIME_MAX_BYTES = 64 * 1024
PRIVATE_RUNTIME = "Application runtime file must be private to this user"


def open_lock(path):
    """Open the single-instance lock: created 0o600, never through a
    symlink, never blocking on a FIFO, and only ever a regular file."""
    try:
        lock = open(
            path,
            "ab",
            buffering=0,
            opener=lambda name, flags: os.open(
                name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
            ),
        )
    except OSError as error:
        raise RuntimeError(f"Application lock {path} is unusable ({error.strerror})") from error
    try:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise RuntimeError(f"Application lock {path} is not a regular file")
    except BaseException:
        lock.close()
        raise
    return lock


def read_runtime(path):
    """Read the running instance's runtime file from one descriptor: a
    regular file this user owns, private, at most RUNTIME_MAX_BYTES."""
    try:
        stream = open(
            path,
            "rb",
            buffering=0,
            opener=lambda name, flags: os.open(
                name, flags | os.O_NOFOLLOW | os.O_NONBLOCK
            ),
        )
    except IsADirectoryError as error:
        raise RuntimeError(PRIVATE_RUNTIME) from error
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise RuntimeError(PRIVATE_RUNTIME) from error
        raise
    with stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise RuntimeError(PRIVATE_RUNTIME)
        if info.st_size > RUNTIME_MAX_BYTES:
            raise RuntimeError(f"Application runtime file exceeds {RUNTIME_MAX_BYTES} bytes")
        data = bytearray()
        while len(data) <= RUNTIME_MAX_BYTES:
            chunk = stream.read(RUNTIME_MAX_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
    if len(data) > RUNTIME_MAX_BYTES:
        raise RuntimeError(f"Application runtime file exceeds {RUNTIME_MAX_BYTES} bytes")
    return json.loads(data)


def write_runtime(path, runtime):
    """Publish the runtime file whole: exclusively create an unguessable
    private 0o600 temporary file with open(..., "x") and a C-level
    functools.partial(os.open, mode=0o600) opener inside the cleanup
    guard so FileIO takes ownership of the descriptor with no Python
    bytecode between os.open() and the file object, then rename it over
    the path. A reader never sees a partial file, an older looser file
    is replaced rather than reused, a signal at any point after opening
    closes and removes the temporary file without leaking a descriptor,
    and any existing name (including a symlink) is left alone on
    FileExistsError."""
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(
            temporary,
            "x",
            encoding="utf-8",
            opener=functools.partial(os.open, mode=0o600),
        ) as stream:
            json.dump(runtime, stream)
        os.replace(temporary, path)
    except FileExistsError:
        raise  # O_EXCL refused a path this call did not create; leave it alone
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_parser():
    parser = argparse.ArgumentParser(prog="mlx-omarchy-assistant", description="Local MLX Chat and typed decisions")
    parser.add_argument("--home", type=Path, default=mlx_omarchy_paths.default_data_home())
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-tokens", type=int, help="explicit output allowance; otherwise size it for the task")
    parser.add_argument("--pair", choices=("everyday", "quality", "compact"))
    parser.add_argument("--resume", action="store_true",
                        help="load the saved pair at login and keep both models resident")
    parser.add_argument("--yes", action="store_true", help="approve downloading the explicitly selected pair")
    parser.add_argument("--wake-word", nargs="?", const="hey_jarvis", default=None,
                        help="opt-in wake word listener (default phrase: hey_jarvis); "
                             "approves the one-time pinned openWakeWord model download")
    parser.add_argument("--wake-threshold", type=float,
                        default=wake_word.WAKE_THRESHOLD_DEFAULT,
                        help="wake word score threshold "
                             f"(default {wake_word.WAKE_THRESHOLD_DEFAULT})")
    return parser


def validate_args(parser, args):
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    if args.once and not args.prompt:
        parser.error("--once requires --prompt")
    if args.yes and not args.pair:
        parser.error("--yes requires an explicit --pair")
    if args.resume and args.pair:
        parser.error("--resume uses the saved pair; do not pass --pair")
    if not 0.0 < args.wake_threshold < 1.0:
        parser.error("--wake-threshold must be between 0 and 1")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if os.environ.get("MLX_OMARCHY_TTFT_TRACE") == "1":
        # Diagnostic: SIGUSR1 dumps every thread's stack to stderr so a
        # stuck turn or a busy GPU lock can be attributed from outside.
        # `signal` stays the module-level import: a function-scope import
        # here makes it local for the whole of main(), and the SIGTERM/
        # SIGINT registration below would raise UnboundLocalError on
        # every startup (v0.7.16 release blocker, caught by gate 2).
        import faulthandler

        faulthandler.register(signal.SIGUSR1, file=sys.stderr)
    validate_args(parser, args)
    directory = args.home / "assistant"
    runtime_file = directory / "application.json"
    lock = None
    server = None
    published = None
    thread = None
    stopping = False
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = open_lock(directory / "application.lock")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            runtime = read_runtime(runtime_file)
            request(runtime, "GET", "/api/session")
        else:
            server = AssistantServer(("127.0.0.1", args.port), args.home)

            def interrupt(_signal, _frame):
                nonlocal stopping
                # The first stop signal unwinds main() into the orderly stop;
                # repeats arrive during that stop and must not abort it.
                if not stopping:
                    stopping = True
                    raise KeyboardInterrupt

            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, interrupt)
            runtime = {"port": server.server_port, "cookie": f"{server.cookie_name}={server.session}", "csrf": server.csrf}
            published = runtime_file
            write_runtime(runtime_file, runtime)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            if args.wake_word:
                server.start_wake_word(args.wake_word, args.wake_threshold)
        if args.pair:
            request(runtime, "POST", "/api/setup", {"pair_id": args.pair, "approve_download": args.yes})
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                state = request(runtime, "GET", "/api/status").get("setup", {})
                if state.get("state") == "error":
                    raise RuntimeError(state["message"])
                if state.get("state") == "complete":
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Model setup exceeded twenty minutes; inspect the setup error before retrying")
        if args.resume:
            try:
                request(runtime, "POST", "/api/resume", {})
            except RuntimeError as error:
                if "already active" not in str(error):
                    raise
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                state = request(runtime, "GET", "/api/status").get("setup") or {}
                if state.get("state") == "error":
                    raise RuntimeError(state.get("message", "Saved pair did not resume"))
                if state.get("state") == "absent":
                    print("MLX Chat: no saved pair to resume", file=sys.stderr)
                    return 0
                if state.get("state") == "complete":
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Saved pair did not resume within twenty minutes")
        if args.terminal or args.prompt:
            return terminal(runtime, args.prompt, args.once, args.max_tokens)
        if not args.no_browser:
            launch = request(runtime, "POST", "/api/launch", {})
            webbrowser.open(f"http://127.0.0.1:{runtime['port']}/#token={launch['token']}")
        print(f"MLX Chat is running at http://127.0.0.1:{runtime['port']}/ (open it through the launcher)")
        if server:
            while thread.is_alive():
                thread.join(timeout=1)
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        stopping = True
        print(f"MLX Chat: {error}", file=sys.stderr)
        return 1
    finally:
        stopping = True
        if server:
            stop_server(server, published, serving=thread is not None and thread.ident is not None)
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
