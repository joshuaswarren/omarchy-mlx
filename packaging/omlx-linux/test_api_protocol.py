#!/usr/bin/env python3
"""Dev-box protocol/unit tests for oMLX API-surface rows (A1/A2/A3/A12/A13/A18/A31).

SUPPORTING EVIDENCE ONLY: this host has no mlx wheel (x86, no accelerator), so
these tests exercise the pure-Python layers with mlx stubbed. Real feature runs
happen on M2/jwm1 via api_parity.sh.

Usage: test_api_protocol.py <omlx-src-root> <evidence-output.json>
Exit: 0 iff all tests pass.
"""

import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path


def stub_mlx() -> None:
    """Register stub mlx modules + a fake omlx package that skips __init__."""
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")

    class _Any:
        def __getattr__(self, name):
            raise AttributeError(f"mlx stub: {name} not available on dev box")

    core.Array = _Any
    mlx.core = core
    sys.modules.setdefault("mlx", mlx)
    sys.modules.setdefault("mlx.core", core)

    src = Path(sys.argv[1]).resolve()
    pkgdir = src / "omlx"
    for pkg, sub in (("omlx", pkgdir), ("omlx.api", pkgdir / "api"), ("omlx.mcp", pkgdir / "mcp")):
        m = types.ModuleType(pkg)
        m.__path__ = [str(sub)]
        sys.modules.setdefault(pkg, m)


def main() -> int:
    stub_mlx()
    from omlx.model_profiles import (  # noqa: E402
        InvalidProfileNameError,
        validate_profile_name,
        slugify_profile_api_name,
        ModelProfile,
    )
    from omlx.model_settings import ModelSettings, ModelSettingsManager  # noqa: E402
    from omlx.usage_history import UsageHistory  # noqa: E402
    from omlx.mcp.config import load_mcp_config  # noqa: E402
    from omlx import websearch as ws  # noqa: E402
    from omlx.api.tool_calling import parse_json_output, validate_json_schema  # noqa: E402
    from omlx.api.anthropic_models import MessagesRequest, AnthropicMessage  # noqa: E402
    from omlx.api.anthropic_utils import convert_anthropic_to_internal  # noqa: E402

    results = []

    def check(name, fn):
        try:
            detail = fn()
            results.append({"test": name, "status": "PASS", "detail": detail or ""})
            print(f"PASS: {name} — {detail or ''}")
        except Exception as e:  # noqa: BLE001
            results.append({"test": name, "status": "FAIL", "detail": repr(e)})
            print(f"FAIL: {name} — {e!r}")

    # --- T1 profiles (A13) ---
    def t1():
        from datetime import datetime, timezone
        validate_profile_name("parity")
        try:
            validate_profile_name("bad/name")
            raise AssertionError("invalid name accepted")
        except InvalidProfileNameError:
            pass
        s1 = slugify_profile_api_name("Parity Profile")
        assert s1 and " " not in s1, s1
        assert slugify_profile_api_name("Parity Profile") == s1, "not deterministic"
        now = lambda: datetime.now(timezone.utc)  # noqa: E731
        p = ModelProfile(name="p", display_name="p", created_at=now(), updated_at=now(),
                         settings={"temperature": 0.5})
        rt = ModelProfile.from_dict(p.to_dict())
        assert rt.settings == {"temperature": 0.5}
        return f"slug={s1}"
    check("T1 profile name validation + slugify + round-trip (A13)", t1)

    # --- T2 settings manager: alias, profile save/apply/expose (A13) ---
    def t2():
        with tempfile.TemporaryDirectory() as td:
            mgr = ModelSettingsManager(Path(td))
            mgr.set_settings("m1", ModelSettings(model_alias="alias-x"))
            assert mgr.get_settings("m1").model_alias == "alias-x"
            mgr.save_profile(model_id="m1", name="par", display_name="Par",
                             description=None, settings={"temperature": 0.5},
                             expose_as_model=True)
            names = [p["name"] for p in mgr.list_profiles("m1")]
            assert "par" in names, names
            exposed = mgr.list_exposed_profile_models()
            hit = [e for e in exposed if e.get("source_model_id") == "m1"]
            assert hit, exposed
            applied = mgr.apply_profile("m1", "par")
            assert applied.temperature == 0.5, applied
            return f"exposed={hit[0].get('model_id')}"
    check("T2 settings manager alias/profile/expose/apply (A13)", t2)

    # --- T3 usage history sqlite round-trip (A31) ---
    def t3():
        with tempfile.TemporaryDirectory() as td:
            h = UsageHistory(Path(td) / "usage.db", enabled=True)
            h.record(model_id="m1", prompt_tokens=10, completion_tokens=5,
                     cached_tokens=0, prefill_duration=0.1, generation_duration=0.2)
            assert h.flush() is not False
            q = h.query("today")
            models = q.get("models") or []
            if isinstance(models, dict):
                assert models.get("m1"), models
                assert models["m1"][0] >= 1, models
            else:
                hit = [m for m in models if (m.get("model_id") if isinstance(m, dict) else m) == "m1"]
                assert hit, models
            con = sqlite3.connect(f"file:{Path(td)/'usage.db'}?mode=ro", uri=True)
            n = con.execute("SELECT count(*) FROM model_usage_hourly").fetchone()[0]
            assert n >= 1, n
            return f"rows={n} qkeys={sorted(q)}"
    check("T3 usage history record/flush/query (A31)", t3)

    # --- T4 MCP config loader (A18) ---
    def t4():
        repo_example = Path(sys.argv[1]) / "mcp.example.json"
        cfg = load_mcp_config(repo_example)
        assert "filesystem" in cfg.servers and cfg.servers["filesystem"].transport == "stdio"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mcp.json"
            p.write_text(json.dumps({"servers": {"echo": {
                "transport": "stdio", "command": "python3",
                "args": ["echo_srv.py"], "enabled": True, "timeout": 30}}}))
            c2 = load_mcp_config(p)
            assert c2.servers["echo"].enabled is True
        return f"example servers={sorted(cfg.servers)}"
    check("T4 MCP config load + validate (A18)", t4)

    # --- T5 websearch provider plumbing (A31) ---
    def t5():
        assert {"ddgs", "brave", "searxng"} <= set(ws.SUPPORTED_PROVIDERS), ws.SUPPORTED_PROVIDERS
        pay = ws.failure_payload("x_code", "boom")
        assert pay["ok"] is False and pay["error"]["code"] == "x_code"
        err = ws._error_from_status(401, "test")
        assert err.needs_user_action is True and err.code == "invalid_authentication"
        try:
            ws.convert_html_to_markdown("<h1>Head</h1><p>Body text</p>")
            raise AssertionError("html->markdown should refuse without optional dep")
        except RuntimeError as e:
            assert "MarkItDown is not installed" in str(e), e
        return "providers + error payloads + honest-refuse without optional dep ok"
    check("T5 websearch providers/errors/markdown (A31)", t5)

    # --- T6 structured output: schema validation + parse (A18) ---
    def t6():
        schema = {"type": "object",
                  "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                  "required": ["name", "age"], "additionalProperties": False}
        rf = {"type": "json_schema", "json_schema": {"name": "person", "schema": schema}}
        okv, err = validate_json_schema({"name": "Ada", "age": 36}, schema)
        assert okv, err
        okv2, _ = validate_json_schema({"name": "Ada", "age": "36"}, schema)
        assert not okv2, "invalid payload passed validation"
        txt, obj, found, perr = parse_json_output('{"name": "Ada", "age": 36}', rf)
        assert found and obj == {"name": "Ada", "age": 36}, (txt, obj, found, perr)
        txt2, obj2, found2, _ = parse_json_output('```json\n{"name": "Ada", "age": 36}\n```', rf)
        assert found2 and obj2["age"] == 36, (txt2, obj2, found2)
        return "valid/invalid/fenced all handled"
    check("T6 json_schema validation + output parsing (A18)", t6)

    # --- T7 Anthropic -> internal request mapping (A2) ---
    def t7():
        req = MessagesRequest(
            model="m1", max_tokens=64,
            messages=[AnthropicMessage(role="user", content="hello")],
            system="be brief",
        )
        msgs = convert_anthropic_to_internal(req)
        assert isinstance(msgs, list) and msgs, msgs
        assert any(m.get("role") in ("user", "system") for m in msgs), msgs
        texts = json.dumps(msgs)
        assert "hello" in texts and "be brief" in texts, texts[:200]
        return f"{len(msgs)} internal messages"
    check("T7 anthropic request mapping (A2)", t7)

    failed = [r for r in results if r["status"] != "PASS"]
    out = {"host": "dev-box (no mlx wheel; stubbed mlx)",
           "evidence_class": "supporting-only", "results": results,
           "pass": len(results) - len(failed), "fail": len(failed)}
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=2))
    print(f"SUMMARY: pass={out['pass']} fail={out['fail']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
