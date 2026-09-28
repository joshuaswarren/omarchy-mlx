"""Launch the local application or attach a terminal to the same coordinator."""

from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .server import AssistantServer


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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlx-omarchy-assistant", description="Local MLX Chat and typed decisions")
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("MLX_OMARCHY_HOME", Path.home() / ".local/share/mlx-omarchy")))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-tokens", type=int, help="explicit output allowance; otherwise size it for the task")
    parser.add_argument("--pair", choices=("everyday", "quality"))
    parser.add_argument("--resume", action="store_true",
                        help="load the saved pair at login and keep both models resident")
    parser.add_argument("--yes", action="store_true", help="approve downloading the explicitly selected pair")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    if args.once and not args.prompt:
        parser.error("--once requires --prompt")
    if args.yes and not args.pair:
        parser.error("--yes requires an explicit --pair")
    if args.resume and args.pair:
        parser.error("--resume uses the saved pair; do not pass --pair")
    directory = args.home / "assistant"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (directory / "application.lock").open("a")
    runtime_file = directory / "application.json"
    server = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if runtime_file.is_symlink() or runtime_file.stat().st_mode & 0o077:
                raise RuntimeError("Application runtime file must be private to this user")
            runtime = json.loads(runtime_file.read_text())
            request(runtime, "GET", "/api/session")
        else:
            server = AssistantServer(("127.0.0.1", args.port), args.home)
            runtime = {"port": server.server_port, "cookie": f"{server.cookie_name}={server.session}", "csrf": server.csrf}
            fd = os.open(runtime_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(runtime, stream)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def interrupt(_signal, _frame):
                raise KeyboardInterrupt

            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, interrupt)
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
        print(f"MLX Chat: {error}", file=sys.stderr)
        return 1
    finally:
        if server:
            server.shutdown()
            server.server_close()
            runtime_file.unlink(missing_ok=True)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
