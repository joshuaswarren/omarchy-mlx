#!/usr/bin/env python3
"""Render pinned oMLX admin templates against empty local API responses for visual checks."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omlx-src", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    admin = args.omlx_src.resolve() / "omlx" / "admin"
    templates = Environment(
        loader=FileSystemLoader(admin / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    locale_dir = admin / "i18n"
    locales = {path.stem.lower(): path for path in locale_dir.glob("*.json")}
    templates.globals["static"] = lambda path: "/admin/static/" + path
    static = (admin / "static").resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print(fmt % values, flush=True)

        def reply(self, body, content_type="application/json", status=200):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path.startswith("/admin/static/"):
                file = (static / path.removeprefix("/admin/static/")).resolve()
                if not file.is_relative_to(static) or not file.is_file():
                    return self.reply(b"not found", "text/plain", 404)
                kind = "text/css" if file.suffix == ".css" else "application/javascript" if file.suffix == ".js" else "image/svg+xml" if file.suffix == ".svg" else "application/octet-stream"
                return self.reply(file.read_bytes(), kind)
            page = {"/": "login.html", "/admin": "login.html", "/admin/": "login.html", "/admin/dashboard": "dashboard.html", "/admin/chat": "chat.html"}.get(path)
            if page:
                preferred = self.headers.get("Accept-Language", "en").split(",", 1)[0].split(";", 1)[0].lower()
                candidates = [preferred, preferred.split("-", 1)[0], "en"]
                lang = next((value for value in candidates if value in locales), "en")
                translations = json.loads(locales[lang].read_text())
                html = templates.get_template(page).render(
                    api_key_configured=True,
                    api_key="",
                    current_lang=lang,
                    locale_json=json.dumps(translations, ensure_ascii=False),
                    t=lambda key, **_kwargs: translations.get(key, key),
                )
                return self.reply(html, "text/html; charset=utf-8")
            if path.startswith("/admin/api/") or path.startswith("/v1/"):
                return self.reply("[]")
            return self.reply(b"not found", "text/plain", 404)

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            size = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(size)
            print(f"STUB API {self.command} {path}", flush=True)
            body = '{"success":true}' if path.endswith("/api/login") else "[]"
            return self.reply(body)

    print(f"Template source: {admin}", flush=True)
    print(f"Listening on http://{args.host}:{args.port} (stub responses; no model or auth)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
