# oMLX admin UI stub-render receipt

## Scope

Device-independent rendering/API-routing evidence for oMLX v0.7.0 (`4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`) only. This is not a real server, authentication, model-loading, chat-generation, downloader, integration, or benchmark run. The dev box has no installed MLX module; `python3` import returned `ModuleNotFoundError`.

## Method

`packaging/omlx-linux/admin_ui_stub.py` rendered the pinned upstream Jinja templates and served their real static assets. It returned empty JSON arrays for `/admin/api/*` and `/v1/*`; `POST /admin/api/login` returned a stub success. The admin pages were viewed in headless Chromium. The dashboard showed no horizontal overflow at viewport widths 375, 768, 1024, and 1440 CSS pixels. Captures are in `screenshots/`.

The login template also rendered Korean text and set the document language to `ko` at 375 CSS pixels using the pinned `ko.json` locale file. This tests the template/i18n resource path, not every supported locale.

Observed UI surfaces: login page; dashboard/status; model manager and HF downloader; benchmark panel with its PP/TG settings and Run Benchmark control; chat screen; integration settings. Stub requests reached login and the dashboard API routes. Distinct dashboard API GET paths returned HTTP 200 from the stub: `/admin/api/{bench/active,bench/accuracy/results,bench/accuracy/queue/status,bench/context/active,device-info,global-settings,models,profile-fields,server-info,stats,update-check,usage}`, plus `/admin/api/hf/{models,tasks}` and `/admin/api/oq/tasks`.

The mock has no model inventory and no authentication/session implementation. Chat therefore remained behind its API-key prompt, the model list was empty, and the benchmark could not start. The dashboard also emitted Alpine/JavaScript reference errors in controls depending on server-side state; these are not classified as oMLX defects from this mocked setup. No timing values or completion output were generated.

## Acceptance status

- Layout/template loading: observed on stub-rendered pages; no overflow at four viewport widths.
- i18n: English and Korean login strings rendered using pinned resource files; broader locale coverage remains untested.
- Login wiring: stub request reached `POST /admin/api/login` and received HTTP 200. Not real authentication.
- Benchmark view/API calls: panel rendered and called active/result endpoints. No model available; no benchmark run; no PP/TG result and no cached-token-warning check.
- A16 and A17: remain OPEN pending the required real M2/jwm1 workflow. This receipt is supporting UI-only evidence.

## Artifacts

- `screenshots/login-{375,768,1024,1440}.png` and `screenshots/login-ko-375.png`
- `screenshots/dashboard-{375,768,1024,1440}.png`
- `screenshots/benchmark-1440.png`
- `screenshots/chat-1440.png`
- `screenshots/integrations-1440.png`
- Matching private copies: `~/.local/share/apple-silicon-lab/artifacts/OmlxAdmin/20261005-admin-ui-stub/`
