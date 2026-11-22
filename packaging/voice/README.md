# Voice packaging skeleton (VXP-3)

Install layout for the offline voice assistant, staged for the
`omarchy-mac-ml` recipe in omarchy-pkgs. Every path is final: what a file
named `usr/...` in the recipe installs is exactly the file here.

```
/usr/bin/omarchy-voice                    <- omarchy-voice        (755)
/usr/bin/omarchy-voice-setup              <- omarchy-voice-setup  (755)
/usr/lib/systemd/user/omarchy-voice.service <- omarchy-voice.service (644, staged never-enabled)
/usr/lib/omarchy-mlx/venv/                <- provided by the omarchy-mlx package; not staged here
```

- `omarchy-voice` runs `mlx_omarchy_assistant` from the omarchy-mlx system
  venv at `/usr/lib/omarchy-mlx/venv` — the same venv every `mlx-omarchy-*`
  launcher uses (name table: `packaging/paths.sh`, `SYSTEM_PREFIX` +
  `VENV_DIR`). No second venv is created.
- `omarchy-voice-setup` is the separate, explicit model-download step — the
  only step that ever needs a network. It approves and fetches the pinned
  pair (`--pair everyday|quality|compact`, flags verified against
  `serve/mlx_omarchy_assistant/__main__.py`) into the user data home.
- `omarchy-voice.service` is a `systemd --user` unit: opt-in, staged but
  never enabled by the package; the user runs
  `systemctl --user enable --now omarchy-voice.service` after setup.
  `After=sound.target` because the pipeline needs PipeWire; the chat unit
  convention (`WantedBy=default.target`, `Restart=on-failure`) is kept.
- Ships no models. After `omarchy-voice-setup` the runtime is fully
  offline, per the post-13362 ML packaging plan (opt-in, never in the
  default image).

Recipe-side consumers: `pkgbuilds/omarchy-mac-ml/` in omacom/omarchy-pkgs
(draft PR #3274 stages these paths with identical names and flags).

## Lint

```
shellcheck omarchy-voice omarchy-voice-setup        # clean (2026-11-22)
systemd-analyze verify <staged>/usr/lib/systemd/user/omarchy-voice.service
# clean on systemd 262 with ExecStart resolved against the staged
# /usr/bin/omarchy-voice (verify checks that ExecStart exists)
```
