# Serve model qualification contract

Owner: ModelCatalogQualification lane, 2026-09-20. Schema owner:
`serve/mlx_omarchy_serve/catalog.py` (single strict validator). This file
defines what flips `qualification.generation` / `qualification.http` to
`qualified` for an entry in `serve/mlx_omarchy_serve/catalog.json`. Nothing
in the refresh path can flip it: `tools/refresh_serve_catalog.py` writes
`availability.*` only, and every promotion is a manual commit.

## Lifecycle rules

1. A pinned `revision` is retained until requalification. The refresher
   reports upstream drift (exit 3) and never moves the pin.
2. Repinning to a new upstream revision is a manual commit that resets the
   entry's qualification to `untested` in the same commit.
3. `recommended` requires `qualification.generation.status == "qualified"`
   and a pinned revision; the validator enforces both. Owner directive
   2026-09-20 is stricter and governs the seed: every recommendation stays
   `false` until HTTP serving is qualified on devices for that exact
   runtime and revision. Until then the catalog recommends nothing.
4. Generation and HTTP are qualified separately. A generation pass says
   nothing about HTTP serving, and vice versa.
5. Implemented backend primitives are not model qualification. "Affine
   bits 2 / group 128 is implemented" is a capability fact, not a pass.

## Evidence rules (repo-wide, from AGENTS.md)

Every device claim needs a receipt file plus a provenance line
(`scripts/mlx_provenance.py`) naming the exact wheel build. Hardware
windows follow the standing discipline: independent `/tmp/m1-gpu.lock`,
LiteLLM exclusion, bounded timeouts, one new failure mode at a time. No
CPU tensor dispatches on the serving path. Numerical contracts compare
against pinned digests, not eyeballed coherence alone.

## Per-entry bars

### qwen3.8-27b-4bit (`mlx-community/Qwen3.8-27B-4bit` @ `10c35caa…`)

- generation: **qualified** 2026-09-20 — text-only CLI smoke on t6021-test-host
  (T6021), candidate wheel `a1251aaa`, mlx-vlm 0.7.1; receipt
  `receipts/2026-09-20-qwen38-text-install/receipt.json` (landed on
  origin/main at `5b183060`; not in the base this branch grew from).
  Limits recorded there carry: not vision, not ANE, not 16 GiB.
- http: **untested**. Bar: serve this exact revision through a server on a
  qualified host; `/v1/models` lists it, `/v1/chat/completions` returns a
  completion id and nonempty assistant content on the pinned prompt set,
  zero CPU tensor dispatches in the trace, digest-stable greedy output
  across two runs. ServePerformanceAudit's t6001-test-host arm (authorized 2026-09-20,
  no duplicate download) is the staged run.

### ternary-bonsai-2-27b-mlx-2bit (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `3f926b41…`)

- generation: **untested**. Bar: (1) the pack's shipped `runtime/*.py` is
  reviewed and accepted by the owner — it is remote code and the catalog
  tooling never executes it; (2) the reused historical specialized loader
  (`mlx_vlm gdn_sink` lineage, F7 fix) loads THIS pinned revision on a
  current-main wheel — historical passes predate recorded pack revisions,
  so the receipt must record the exact pack revision this time; (3) 96-step
  greedy window is coherent with `nan_at` null and compile-on digest
  stability across two reps. Conversion to a stock loader is out of scope
  until the custom model type is upstreamed or ported.
- http: **untested**; only meaningful after generation passes.

### qwen3.8-35b-a3b-distill (`empero-ai/Qwen3.8-35B-A3B-Distill` @ `bcc2dbe2…`)

- generation: **untested**. Bar: (1) a Q4/group-64 MLX conversion built
  FROM the pinned bf16 revision (preserving tokenizer and chat template),
  artifact sha256 recorded; conversion host inspected first (this dev box:
  48 GB RAM / 117 GB free disk / 16 cores — a full 72 GiB snapshot plus
  ~20 GiB output fits with cleanup, streaming preferred); (2) the
  `qwen3_5_moe` MoE graph (256 experts, 8 active, shared expert, gdn
  hybrid attention) loads and evaluates on the Omarchy backend with zero
  CPU dispatches; (3) memory gate passes on a 64 GiB host against the
  converted artifact's real weights; (4) 32-token contract versus a
  pinned reference with fixed digest pins. The bf16 original is NOT a
  servable target anywhere in the fleet (66.97 GiB weights > 64 GiB).
- http: **untested**; only after generation passes.

### laya-mlx (`aac6fef/laya-mlx` @ `20aed815…`)

- Owned by the Laya serving lane (isolated, no overlap). Catalog contract:
  qualification flips only on that lane's receipt naming the pack revision
  and the runtime path used. The pack is not mlx_lm loadable (custom
  `laya-mlx` format). Source drift is recorded: pack manifest pins
  `convaiinnovations/laya` @ `c5d78730…`, source head has moved to
  `1c5edc17…` — a repack invalidates and restarts qualification.

## Memory semantics

Admission is a per-machine runtime decision. Live `MemAvailable` is ground
truth: already-resident work reduces it without any cooperation, explicit
reservations (`mlx-omarchy-serve reserve`) cover declared co-residents
before they launch, and per-model KV/workspace ride in the estimate. There
is no cross-machine accounting by design.

`capability.min_mem_gib` is a static weights floor:
`ceil(weights_bytes * 1.25 / 2^30)`, documented per entry. It is a
rejection floor, never a fitness claim. Entries without a measured
`kv_bytes_per_token` get the budget module's labeled flat margin (2 GiB);
a measured `peak_estimate_bytes` overrides when it lands. The CLI worker
and the serve flow refuse a model whose floor exceeds available memory,
naming the model, the floor, and what is available; refusal is data, not
failure of the model.

Context enforcement note (ServeCatalogImplementation, 2026-09-20): on the
mlx-lm path, prompt + output tokens are enforced against the admitted
context by the project's shim — upstream `--max-tokens` is only a
generation default, not a context cap. Any doc that states a context cap
should reference the shim behavior, not upstream flags.

Measured fleet facts (2026-09-20, Main): the M2 Max host reports
~94.25 GiB OS-usable and ~80.77 GiB available; the user fleet has no
single memory cap (64 GiB t6001-test-host, 96 GB-class M2, 16 GiB m1-test-host pending).
Against those numbers the 35B bf16 floor (84 GiB) exceeds current
measured availability, so it refuses today on the M2 while remaining a
legitimate entry for a quieter host state or a smaller variant.

## Variants

Each quantization variant is its own catalog entry (own repo, pinned
revision, quant, memory facts, and generation/http qualification) grouped
by a first-class `family` key, ordered by curated `priority` within kind.
`recommended` sits on exactly one entry per catalog — the top curated
chat variant. Metadata-available and qualified stay distinct per variant:
an available 8-bit download is not a qualified 8-bit runtime. There is no
blanket Q4 assumption; Q4/Q5/Q6/Q8/BF16 and fp-format variants are
recorded as they verifiably exist upstream.

## Qualification gaps (open, 2026-09-20)

- Qwen3.8-27B-4bit: HTTP serving unqualified; 16 GiB host gates unrun.
- Bonsai-2-27B: no receipt pins the pack revision; pack runtime code
  unreviewed; nothing qualified on a released wheel.
- Qwen3.8-35B-A3B-Distill: no MLX Q4 conversion exists; `qwen3_5_moe`
  loader path unproven on this stack; no device run.
- laya-mlx: runtime adaptation in flight in its own lane; nothing run.
- The Qwen3.8 qualification receipt exists on origin/main (`5b183060`)
  but not on this branch's base (`10b2f242`); integration must merge it
  before the catalog's qualified claim and the receipt coexist in one tree.
