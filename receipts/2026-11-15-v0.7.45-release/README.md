# omarchy-mlx v0.7.45 release receipt

[Joshua's agent]

Tag `v0.7.45` points at `2e49e2ca5a34059f1b956f84419e6477314d45cf` ("speculative verify: never write MLX_OMARCHY_SDPA_SPLIT_ROWS (#456)"). `RELEASE-NOTES.md` is the published body. The release was drafted first, every gate ran against the draft's assets, and it was published only after all of them passed (published 2026-11-15 12:46 UTC).

## Assets (SHA256SUMS of the release)

```
934ea621c7145af7c3a1c6bedc5de8012b19faf5580ade283d8346a99dce44f8  mlx_omarchy-0.32.4.dev202611151053+2e49e2ca-cp311-cp311-linux_x86_64.whl
8f61bf82d4604b712cf2283b137324af742368387782d265827de9537146cf58  mlx_omarchy-0.32.4.dev202611151126+2e49e2ca-cp314-cp314-linux_aarch64.whl
2aa4571f0258de280de64292a6842fda404b54f8ea1c1220482bb3a62167b1df  omarchy-mlx-vendor-wheels-v0.7.45-cp314-aarch64.tar
845033e5b78bec614b3124d3ffc2534488b60b822bdee1141e66b36e5d74b621  omarchy-mlx-vendor-wheels-v0.7.45-cp314-aarch64.tar.sha256
8f61bf82d4604b712cf2283b137324af742368387782d265827de9537146cf58  omarchy-mlx-wheel-cp314-aarch64.whl
```

- `scripts/verify-release-assets.py v0.7.45` (the script from the tag) on the uploaded draft printed: VERIFIED: every uploaded asset matches what the release claims. The script was re-run after the notes were updated and before publishing.
- The x86_64 wheel was built on the x86 build container at the pin; the aarch64 wheel and vendor tar were built on the 13-inch M1.
- An earlier x86_64 build at a previous pin (`7845b35e`) was discarded when #456 changed the pin; no asset from it was uploaded.

## Gate record

All times UTC, 2026-11-15, from the `gates.done` file of the run. Machine names are left out on purpose. Everything ran on the 13-inch M1 (G13G), from the tag's tree against the draft assets, through the lab's GPU queue.

```
g1-clean-install_RC=0 12:21:46
g3-online-4b-card_RC=0 12:24:09
g6-codec_RC=0 12:29:20
g8-kokoro_RC=0 12:29:31
g7a-packaged-icd_RC=0 12:29:31
g7b-system-install_RC=0 12:29:52
g9-speak-queue_RC=0 12:30:04
g7c-ane-worker-verify_RC=0 12:30:36
g7d-fresh-transcribe_RC=0 12:31:21
g13-gdn-maskless_build_RC=0 12:42:15
g13-gdn-maskless_run_RC=0 12:44:26
g15-trig_build_RC=0 12:44:34
g15-trig_run_RC=0 12:44:34
```

- g13 and g15 build from the wheel's build tree, which exists only on the host that built the aarch64 wheel. For this release that host was the 13-inch M1, so they ran there. A first attempt queued on the 14-inch M2 Max refused ("build tree missing; build the wheel first") before any test ran, and was not used. No gate ran on the 14-inch M2 Max or the 16-inch M1 Max for this release.
- Pinned decode on the uploaded aarch64 wheel, 13-inch M1, Qwen3.8-2B 4-bit, greedy, seed 0, 128 tokens requested: generated-ids digest `sha256:775bf034e185c8f9` in 5 of 5 runs; 28.55, 28.68, 28.62, 28.53, 28.60 tok/s, median 28.60. Identity check only; no older wheel was run on the same bench.

## Known issue carried forward

The GPU timeouts on the 16-inch M1 Max (G13C) are not understood. The release notes say so and ship the split-rows route off. A dispatch trace of one faulting process showed the fault arriving after the last submit had completed, at or just after process exit; that evidence is on linux-aurora #335. A synchronize-and-wait test is queued.
