# Card promotion receipt, 2026-09-30

Cards on the default pairs without the model emitting JSON: when a chat turn
ends without a valid `assistant-ui` fence, `card_promotion.extract_text`
builds at most one card from the reply's markdown. On 8a1e25843 the pairs
became Everyday = Qwen3.5-9B, Compact = Qwen3-4B-Instruct-2507, Quality =
Qwen3.8-27B (the 2B left the catalog).

**Card gates: all three pairs pass HELD-OUT v4 in the config the product
ships, and the 9B and 4B pass first-text latency.** This receipt does not mark
any pair qualified (the pair gates also cover voice, memory, routing and more;
see `docs/serve.md`). `extension.card_format` stays unset on every catalog
entry, as the pre-registered rule requires (below).

## Kernel of every row

| Rows | Kernel | Boot |
|---|---|---|
| v1, v2, finite-logits probe | polltx (7.1.13-ARCH-polltx) | several, 2026-09-29/30 |
| v3 on 2B, 9B, 27B, 4B; first latency attempt | polltx | 95675db4 |
| Stock baseline below; fenced-json v3, v4, final latency | stock 7.1.13-3-1-ARCH | 4aed18b3, b68db721, 0e2c3743 (USB chain load); 10fba2de, 40f95214, 529d10a1 (disk boot) |

Card counts are model behaviour under greedy decoding and are compared across
kernels; timings are not. The same-microbench baseline shows why:

| 9B, ModelBench perf phase (06964b800) | polltx (boot 2a4f18f7) | stock (boot 4aed18b3) |
|---|---|---|
| first token, 244-token prompt | 0.779 s | 0.805 s |
| prefill 512 tokens | 1.489 s | 1.644 s |
| prefill 2048 tokens | 5.110 s | 6.212 s |
| decode after 512 tokens | **26.0 tok/s** | **16.7 tok/s** (min 13.1, max 24.0) |
| decode after 2048 tokens | 21.7 tok/s | 20.7 tok/s |

Decode after a 512-token prompt is 36% slower on the stock kernel and much
noisier. Prefill is 10-22% slower. No latency number from the polltx kernel
may be compared with a stock-kernel number. Both rows use wheel
0.32.3.dev202609291615+06711ad (provenance `verified=match`). Raw:
`artifacts/baseline_perf_qwen3.5-9b_stock.json`,
`../2026-09-30-chat-model-bench/v3/qwen3.5-9b.json`.

## Gates per pair (thresholds fixed before each suite ran)

| Gate | Everyday 9B | Compact 4B | Quality 27B |
|---|---|---|---|
| v3 default config: >= 15/18 cards | **14/18 FAIL** | 17/18 pass | 15/18 pass |
| v3 default config: 0/18 spurious | 0/18 pass | 0/18 pass | 0/18 pass |
| v3 fenced-json config (selection only, stock kernel) | 21/36 run: 17/18 cards, **1 spurious: FAIL** | 18/18, 0/18 pass | not run (default passed) |
| config chosen by the pre-registered rule | default | fenced-json | default |
| **v4, chosen config** (sha256 `7f807008...`) | **16/18, 0/18 spurious: PASS** | **17/18, 0/18: PASS** | **18/18, 0/18: PASS** |
| first-text p95, chosen config, stock kernel | **1049 ms: PASS** | **2398 ms: FAIL** | not required |
| v4, 4B default config (run after the fenced one) | - | **18/18, 0/18: PASS** | - |
| first-text p95, 4B default config | - | **656 ms: PASS** | - |
| every card passes `validate_components` | yes (server-side) | yes | yes |
| `extension.card_format` | unset (default chosen) | unset (fenced failed latency) | unset (default chosen) |

What the product does now, with `card_format` unset everywhere: no card schema
on ordinary chat turns, markdown promotion for cards. That is the config each
pair passed in: 9B v4 16/18 with p95 1.05 s, 4B v4 18/18 with p95 0.66 s,
27B v4 18/18. The 27B's latency was not part of this gate.

The old Everyday 2B ran v3 first: 18/18 cards but **1/18 spurious**
(v3-29), so it failed too.

v3 is spent for the current rules: its misses drove the headed-section
comparison rule and the v3-29 fix. From here v3 only selects a prompt config;
v4 certifies.

## What changed in the product

Rules (all in `serve/mlx_omarchy_assistant/card_promotion.py`, docstring and
`docs/serve.md` describe them):

1. A card needs a request for that artifact in the user's message and a
   matching structure in the reply (25ef46bcd). Fixed v1/v2 spurious cards
   from explanatory tables and model fences on plain prompts.
2. Ordinary chat sends no card schema (25ef46bcd). The full schema goes to
   Laya turns, cue words markdown cannot carry, and catalog entries with
   `card_format: "fenced-json"`.
3. Comparisons from two or more headed sections of labelled bullets
   (12dc67183): the 9B and 27B wrote requested comparisons that way (v3-05,
   06, 08). "contrast" and "compares" count as requests.
4. Topic words ("phases", "milestones", "highlights") do not count when the
   message leads with an explanation, and "without using a list" / "not as
   a list" / "no bullet points" turn promotion off (12dc67183; v3-29 on the
   2B).
5. Every pattern is linear; eight 1 MiB hostile inputs each finish under
   2 s in the tests. The v1 parser did not finish one 200 KB line in 120 s.

Coordinator bugs found by the M2 runs (fa3865a7b):

- `extension.card_format` was read from `catalog["entries"]`; the key is
  `models`, so fenced-json could never take effect. The first fenced-json
  9B run returned 18 of 18 replies byte-identical to the default run and is
  set aside (`invalid_*` on the M2, not scored).
- The component repair call sent only a system message. Qwen 3.5's chat
  template raises `TemplateError: No user query found in messages`, so any
  invalid fence on the 9B ended the turn in an error (v3-15 on the 9B:
  "Three quick facts about the Eiffel Tower"). The rejected block now goes
  as the user turn.

Runner defects fixed along the way (receipt tooling only): heartbeats from a
thread (v2's 27B lost 14 turns to the 20 s heartbeat cancel), setup deadline
arithmetic, fresh pair homes, SSE event parsing, developer-qualification env
for the untested pairs, a pinned 4B snapshot missing `README.md` (downloaded
once online), per-row kernel and boot id.

## Latency: the first attempt (polltx) is not valid

`artifacts/latency_polltx_cancel_*.json` (polltx, boot 95675db4, alone on
the GPU: only the probe's own ticket in the process list):

| | p50 | p95 |
|---|---|---|
| 9B | 3554 ms | 6252 ms |
| 4B | 7667 ms | 9512 ms |

That probe cancelled each turn right after its first text. The 9B p50 is the
speech-yield probe timeout (3.0 s) plus about half a second, which suggests
each turn waited on a worker still decoding the cancelled reply; this is an
inference, not verified. The probe now lets every turn finish (max_tokens
64) before the next starts (ef64d12e7). The gate needs a fresh stock-kernel
run; these numbers do not pass or fail it.

## HELD-OUT v4 (frozen, stock kernel)

`tests/fixtures/cards_held_out_v4.json`, sha256
`7f8070083407c9e3b7f274f8a4104316184805402ed09fc436acfa046ac2b29d`, frozen in
41817c8af before any model or `extract_text` saw it. 18 card-worthy, 9 plain,
9 near-miss (prose requests including "without using a list", short lists,
code, explanations likely to use headed sections). Card code: origin/main
fa3865a7b (card_promotion `06e42942...`, both coordinator fixes). Kernel
7.1.13-3-1-ARCH (disk boots 10fba2de, 40f95214, 529d10a1); wheel
0.32.3.dev202609291615+06711ad. max_tokens 700, real HTTP turns, one run per
pair and config; after a reboot the next ticket resumed from the per-prompt
checkpoint and no prompt was rerun. The plan was pre-registered in the
notebook before the runs.

Everyday 9B, default config: **16/18 cards, 0/18 spurious, PASS.**
```
    v4-01 card-worthy checklist                  complete   74.3s ['checklist'] promoted pass=True
    v4-02 card-worthy checklist                  complete   40.1s ['checklist'] promoted pass=True
    v4-03 card-worthy checklist                  complete   76.2s ['checklist'] promoted pass=True
    v4-04 card-worthy checklist                  complete   17.0s ['checklist'] promoted pass=True
    v4-05 card-worthy comparison                 complete   91.2s ['comparison'] promoted pass=True
    v4-06 card-worthy comparison                 complete   72.1s ['comparison'] promoted pass=True
    v4-07 card-worthy comparison                 complete   71.1s [] -        pass=False
    v4-08 card-worthy comparison                 complete   83.2s ['comparison'] promoted pass=True
    v4-09 card-worthy timeline                   complete   85.2s ['timeline'] promoted pass=True
    v4-10 card-worthy timeline                   complete   84.2s ['timeline'] promoted pass=True
    v4-11 card-worthy timeline                   complete   88.2s ['timeline'] promoted pass=True
    v4-12 card-worthy timeline                   complete   86.4s [] -        pass=False
    v4-13 card-worthy facts                      complete   31.1s ['facts'] fence    pass=True
    v4-14 card-worthy facts                      complete    7.0s ['facts'] promoted pass=True
    v4-15 card-worthy facts                      complete   70.1s ['facts'] fence    pass=True
    v4-16 card-worthy mixed                      complete   85.2s ['comparison'] promoted pass=True
    v4-17 card-worthy mixed                      complete   87.2s ['checklist'] promoted pass=True
    v4-18 card-worthy mixed                      complete   43.1s ['facts'] promoted pass=True
    v4-19 plain       none                       complete   76.1s [] -        pass=True
    v4-20 plain       none                       complete   86.2s [] -        pass=True
    v4-21 plain       none                       complete    7.0s [] -        pass=True
    v4-22 plain       none                       complete   90.2s [] -        pass=True
    v4-23 plain       none                       complete   91.2s [] -        pass=True
    v4-24 plain       none                       complete   58.3s [] -        pass=True
    v4-25 plain       none                       complete   67.1s [] -        pass=True
    v4-26 plain       none                       complete   42.1s [] -        pass=True
    v4-27 plain       none                       complete   83.2s [] -        pass=True
    v4-28 near-miss   prose-request              complete   22.0s [] -        pass=True
    v4-29 near-miss   prose-request              complete   18.0s [] -        pass=True
    v4-30 near-miss   prose-request              complete   27.1s [] -        pass=True
    v4-31 near-miss   short-list                 complete    4.0s [] -        pass=True
    v4-32 near-miss   short-list                 complete    2.0s [] -        pass=True
    v4-33 near-miss   code-only                  complete   68.1s [] -        pass=True
    v4-34 near-miss   code-only                  complete   37.1s [] -        pass=True
    v4-35 near-miss   explanation-with-sections  complete   84.2s [] -        pass=True
    v4-36 near-miss   explanation-with-sections  complete   86.2s [] -        pass=True
```
Both misses (v4-07, v4-12) are pipe tables whose separator row ends in a
malformed `| : |` cell; the table rule needs at least one dash per separator
cell and refused them. This is a rule gap found on v4 and left unfixed
(fixing it now would tune on v4); see Follow-ups.

Compact 4B, fenced-json config (the selection's choice): **17/18,
0/18, PASS.**
```
    v4-01 card-worthy checklist                  complete   43.1s ['checklist'] promoted pass=True
    v4-02 card-worthy checklist                  complete   17.0s ['checklist'] promoted pass=True
    v4-03 card-worthy checklist                  complete   28.1s ['checklist'] promoted pass=True
    v4-04 card-worthy checklist                  complete   27.1s ['checklist'] promoted pass=True
    v4-05 card-worthy comparison                 complete   82.2s ['comparison'] promoted pass=True
    v4-06 card-worthy comparison                 complete   52.1s ['comparison'] promoted pass=True
    v4-07 card-worthy comparison                 complete   30.1s [] -        pass=False
    v4-08 card-worthy comparison                 complete   56.1s ['comparison'] promoted pass=True
    v4-09 card-worthy timeline                   complete   78.1s ['timeline'] promoted pass=True
    v4-10 card-worthy timeline                   complete   26.1s ['timeline'] promoted pass=True
    v4-11 card-worthy timeline                   complete  194.4s ['timeline'] fence    pass=True
    v4-12 card-worthy timeline                   complete   33.1s ['timeline'] promoted pass=True
    v4-13 card-worthy facts                      complete    8.0s ['facts'] promoted pass=True
    v4-14 card-worthy facts                      complete    8.0s ['facts'] promoted pass=True
    v4-15 card-worthy facts                      complete   17.0s ['facts'] promoted pass=True
    v4-16 card-worthy mixed                      complete   67.1s ['comparison'] promoted pass=True
    v4-17 card-worthy mixed                      complete  103.2s ['checklist'] promoted pass=True
    v4-18 card-worthy mixed                      complete   25.0s ['facts'] promoted pass=True
    v4-19 plain       none                       complete   21.0s [] -        pass=True
    v4-20 plain       none                       complete   40.1s [] -        pass=True
    v4-21 plain       none                       complete    9.0s [] -        pass=True
    v4-22 plain       none                       complete   34.0s [] -        pass=True
    v4-23 plain       none                       complete   14.0s [] -        pass=True
    v4-24 plain       none                       complete   18.0s [] -        pass=True
    v4-25 plain       none                       complete   65.1s [] -        pass=True
    v4-26 plain       none                       complete   30.1s [] -        pass=True
    v4-27 plain       none                       complete   16.0s [] -        pass=True
    v4-28 near-miss   prose-request              complete   26.0s [] -        pass=True
    v4-29 near-miss   prose-request              complete   22.0s [] -        pass=True
    v4-30 near-miss   prose-request              complete   35.1s [] -        pass=True
    v4-31 near-miss   short-list                 complete    3.0s [] -        pass=True
    v4-32 near-miss   short-list                 complete    3.0s [] -        pass=True
    v4-33 near-miss   code-only                  complete   33.1s [] -        pass=True
    v4-34 near-miss   code-only                  complete   17.0s [] -        pass=True
    v4-35 near-miss   explanation-with-sections  complete   47.1s [] -        pass=True
    v4-36 near-miss   explanation-with-sections  complete   33.1s [] -        pass=True
```

Compact 4B, default config: **18/18, 0/18, PASS.** Run after the fenced
config had seen the same prompts (Main's GO, 2026-10-01), because the fenced
config failed latency and the default is what ships. Rules unchanged; nothing
was tuned on v4.
```
    v4-01 card-worthy checklist                  complete   72.1s ['checklist'] promoted pass=True
    v4-02 card-worthy checklist                  complete   29.1s ['checklist'] promoted pass=True
    v4-03 card-worthy checklist                  complete   70.1s ['checklist'] promoted pass=True
    v4-04 card-worthy checklist                  complete   53.1s ['checklist'] promoted pass=True
    v4-05 card-worthy comparison                 complete   71.1s ['comparison'] promoted pass=True
    v4-06 card-worthy comparison                 complete   73.1s ['comparison'] promoted pass=True
    v4-07 card-worthy comparison                 complete   45.1s ['comparison'] promoted pass=True
    v4-08 card-worthy comparison                 complete   72.1s ['comparison'] promoted pass=True
    v4-09 card-worthy timeline                   complete   71.1s ['timeline'] promoted pass=True
    v4-10 card-worthy timeline                   complete   56.1s ['timeline'] promoted pass=True
    v4-11 card-worthy timeline                   complete   74.1s ['timeline'] promoted pass=True
    v4-12 card-worthy timeline                   complete   38.1s ['timeline'] promoted pass=True
    v4-13 card-worthy facts                      complete    7.0s ['facts'] promoted pass=True
    v4-14 card-worthy facts                      complete    3.0s ['facts'] promoted pass=True
    v4-15 card-worthy facts                      complete   17.0s ['facts'] promoted pass=True
    v4-16 card-worthy mixed                      complete   70.1s ['comparison'] promoted pass=True
    v4-17 card-worthy mixed                      complete   71.1s ['checklist'] promoted pass=True
    v4-18 card-worthy mixed                      complete   37.1s ['facts'] promoted pass=True
    v4-19 plain       none                       complete   16.0s [] -        pass=True
    v4-20 plain       none                       complete   30.1s [] -        pass=True
    v4-21 plain       none                       complete    5.0s [] -        pass=True
    v4-22 plain       none                       complete   20.0s [] -        pass=True
    v4-23 plain       none                       complete   15.0s [] -        pass=True
    v4-24 plain       none                       complete   12.0s [] -        pass=True
    v4-25 plain       none                       complete   58.1s [] -        pass=True
    v4-26 plain       none                       complete   15.0s [] -        pass=True
    v4-27 plain       none                       complete   19.0s [] -        pass=True
    v4-28 near-miss   prose-request              complete   22.0s [] -        pass=True
    v4-29 near-miss   prose-request              complete   14.0s [] -        pass=True
    v4-30 near-miss   prose-request              complete   28.1s [] -        pass=True
    v4-31 near-miss   short-list                 complete    1.0s [] -        pass=True
    v4-32 near-miss   short-list                 complete    8.0s [] -        pass=True
    v4-33 near-miss   code-only                  complete   17.0s [] -        pass=True
    v4-34 near-miss   code-only                  complete   12.0s [] -        pass=True
    v4-35 near-miss   explanation-with-sections  complete   39.1s [] -        pass=True
    v4-36 near-miss   explanation-with-sections  complete   26.1s [] -        pass=True
```

Quality 27B, default config: **18/18, 0/18, PASS.**
```
    v4-01 card-worthy checklist                  complete  182.3s ['checklist'] promoted pass=True
    v4-02 card-worthy checklist                  complete  146.3s ['checklist'] promoted pass=True
    v4-03 card-worthy checklist                  complete  166.3s ['checklist'] promoted pass=True
    v4-04 card-worthy checklist                  complete   40.1s ['checklist'] promoted pass=True
    v4-05 card-worthy comparison                 complete  162.3s ['comparison'] promoted pass=True
    v4-06 card-worthy comparison                 complete  165.3s ['comparison'] promoted pass=True
    v4-07 card-worthy comparison                 complete  136.2s ['comparison'] promoted pass=True
    v4-08 card-worthy comparison                 complete  168.3s ['comparison'] promoted pass=True
    v4-09 card-worthy timeline                   complete  165.3s ['timeline'] promoted pass=True
    v4-10 card-worthy timeline                   complete  128.2s ['timeline'] promoted pass=True
    v4-11 card-worthy timeline                   complete  157.3s ['timeline'] promoted pass=True
    v4-12 card-worthy timeline                   complete  132.2s ['timeline'] promoted pass=True
    v4-13 card-worthy facts                      complete   67.1s ['facts'] fence    pass=True
    v4-14 card-worthy facts                      complete   13.0s ['facts'] promoted pass=True
    v4-15 card-worthy facts                      complete  166.3s ['facts'] fence    pass=True
    v4-16 card-worthy mixed                      complete  159.3s ['comparison'] promoted pass=True
    v4-17 card-worthy mixed                      complete  165.3s ['checklist'] promoted pass=True
    v4-18 card-worthy mixed                      complete   77.1s ['facts'] promoted pass=True
    v4-19 plain       none                       complete  100.2s [] -        pass=True
    v4-20 plain       none                       complete  154.3s [] -        pass=True
    v4-21 plain       none                       complete   17.0s [] -        pass=True
    v4-22 plain       none                       complete  165.3s [] -        pass=True
    v4-23 plain       none                       complete  112.2s [] -        pass=True
    v4-24 plain       none                       complete   79.2s [] -        pass=True
    v4-25 plain       none                       complete  170.3s [] -        pass=True
    v4-26 plain       none                       complete  120.2s [] -        pass=True
    v4-27 plain       none                       complete  162.3s [] -        pass=True
    v4-28 near-miss   prose-request              complete   49.1s [] -        pass=True
    v4-29 near-miss   prose-request              complete   32.1s [] -        pass=True
    v4-30 near-miss   prose-request              complete   34.1s [] -        pass=True
    v4-31 near-miss   short-list                 complete    7.0s [] -        pass=True
    v4-32 near-miss   short-list                 complete    3.0s [] -        pass=True
    v4-33 near-miss   code-only                  complete   28.0s [] -        pass=True
    v4-34 near-miss   code-only                  complete   75.1s [] -        pass=True
    v4-35 near-miss   explanation-with-sections  complete  161.3s [] -        pass=True
    v4-36 near-miss   explanation-with-sections  complete  155.3s [] -        pass=True
```

## First-text latency (stock kernel, alone on the GPU)

`run_latency.py` at ef64d12e7+: 3 warm-up turns, then 30 ordinary prompts with
no card words ("Say something short about the number N"), max_tokens 64, each
turn runs to completion before the next. First text = first non-empty `text`
event on the SSE stream, measured from the POST. p95 is nearest-rank. The
process list before and after holds only this probe's own assistant, chat
worker and Laya worker; provenance `verified: match`.

| Pair, config | p50 | p95 | gate 2000 ms | boot, load before |
|---|---|---|---|---|
| Everyday 9B, default | 1002 ms | 1049 ms | pass | 40f95214, 0.98 |
| Compact 4B, fenced-json | 2364 ms | 2398 ms | **fail** | 40f95214, 0.34 |
| Compact 4B, default | 596 ms | 656 ms | pass | 529d10a1, 0.00 |

The full schema adds about 1.8 s of prefill to every 4B turn. Raw:
`artifacts/latency_stock_*.json`, `artifacts/latency_stockdefault_*.json`.

## `extension.card_format` decision

Rule (pre-registered): set `fenced-json` only for a pair whose chosen config is
fenced-json AND passes v4 AND latency. 9B and 27B chose the default; the 4B
chose fenced-json, which passed v4 but failed latency. So no entry gets the
field, and the catalog is unchanged.

## Follow-ups

- Separator cells without a dash (`| : |`) cost the 9B two v4 cards. A fix
  needs invented DEV cases and a new frozen suite (v5) to certify it.
- The coordinator's speech-yield probe may add a 3 s wait when a turn starts
  while the worker still decodes a cancelled reply. The first latency probe
  suggested it; it was not verified.
- The first stock-kernel ModelBench row shows 9B decode at 16.7 tok/s against
  26.0 on polltx; the kernel team owns that.

## v3 with `card_format: "fenced-json"` (config selection, stock kernel)

Declared in the notebook before it ran. Checkout: the frozen v3 rules
(card_promotion `9f1177f9...`), coordinator fa3865a7b, and the catalog with
`extension.card_format: "fenced-json"` on the 9B and 4B, so every chat turn
carries the full schema and the model's own fence wins, with promotion as the
fallback. A card from either path counts; so does a spurious one. Boots
b68db721 and 0e2c3743, kernel 7.1.13-3-1-ARCH.

Rule (fixed before the run): pick the config that passes v3; if both pass,
the one with more cards; a tie or two failures keep the default.

Compact 4B: 18/18 cards, 0/18 spurious (pass; default 17/18, 0/18) ->
**fenced-json**.
```
    v3-01 card-worthy checklist                complete   33.1s ['checklist'] promoted pass=True
    v3-02 card-worthy checklist                complete   39.1s ['checklist'] promoted pass=True
    v3-03 card-worthy checklist                complete   91.2s ['checklist'] fence    pass=True
    v3-04 card-worthy checklist                complete   28.1s ['checklist'] promoted pass=True
    v3-05 card-worthy comparison               complete   36.1s ['comparison'] promoted pass=True
    v3-06 card-worthy comparison               complete   52.1s ['comparison'] promoted pass=True
    v3-07 card-worthy comparison               complete   36.1s ['comparison'] promoted pass=True
    v3-08 card-worthy comparison               complete  171.3s ['comparison'] promoted pass=True
    v3-09 card-worthy timeline                 complete   99.2s ['timeline'] promoted pass=True
    v3-10 card-worthy timeline                 complete  200.4s ['timeline'] promoted pass=True
    v3-11 card-worthy timeline                 complete   48.1s ['timeline'] promoted pass=True
    v3-12 card-worthy timeline                 complete   13.0s ['timeline'] promoted pass=True
    v3-13 card-worthy facts                    complete   11.0s ['facts'] promoted pass=True
    v3-14 card-worthy facts                    complete    7.0s ['facts'] promoted pass=True
    v3-15 card-worthy facts                    complete   18.0s ['facts'] promoted pass=True
    v3-16 card-worthy mixed                    complete   64.1s ['comparison'] promoted pass=True
    v3-17 card-worthy mixed                    complete  140.3s ['checklist'] promoted pass=True
    v3-18 card-worthy mixed                    complete    7.0s ['checklist'] promoted pass=True
    v3-19 plain       none                     complete   28.1s [] -        pass=True
    v3-20 plain       none                     complete   31.1s [] -        pass=True
    v3-21 plain       none                     complete   38.1s [] -        pass=True
    v3-22 plain       none                     complete   15.0s [] -        pass=True
    v3-23 plain       none                     complete   62.1s [] -        pass=True
    v3-24 plain       none                     complete   19.0s [] -        pass=True
    v3-25 plain       none                     complete   47.1s [] -        pass=True
    v3-26 plain       none                     complete    6.0s [] -        pass=True
    v3-27 plain       none                     complete   35.1s [] -        pass=True
    v3-28 near-miss   list-in-prose            complete   24.0s [] -        pass=True
    v3-29 near-miss   list-in-prose            complete   32.1s [] -        pass=True
    v3-30 near-miss   short-list               complete    4.0s [] -        pass=True
    v3-31 near-miss   short-list               complete    9.0s [] -        pass=True
    v3-32 near-miss   code-only                complete    5.0s [] -        pass=True
    v3-33 near-miss   code-only                complete    5.0s [] -        pass=True
    v3-34 near-miss   table-noun               complete   26.0s [] -        pass=True
    v3-35 near-miss   explanation-with-bullets complete   16.0s [] -        pass=True
    v3-36 near-miss   history                  complete   25.1s [] -        pass=True
```

Everyday 9B: stopped at 21/36 when the GPU slot closed; already 1 spurious
(v3-19 "Why do cats purr?" got a model-fenced facts card), so it cannot pass.
Neither config passes -> **default**.
```
    v3-01 card-worthy checklist                complete  175.5s ['checklist'] fence    pass=True
    v3-02 card-worthy checklist                complete  107.2s ['checklist'] fence    pass=True
    v3-03 card-worthy checklist                complete  137.2s ['checklist'] promoted pass=True
    v3-04 card-worthy checklist                complete   96.2s ['checklist'] promoted pass=True
    v3-05 card-worthy comparison               complete  124.2s ['comparison'] promoted pass=True
    v3-06 card-worthy comparison               complete   57.1s ['comparison'] fence    pass=True
    v3-07 card-worthy comparison               complete   86.1s ['comparison'] fence    pass=True
    v3-08 card-worthy comparison               complete  180.6s [] -        pass=False
    v3-09 card-worthy timeline                 complete  145.3s ['timeline'] fence    pass=True
    v3-10 card-worthy timeline                 complete   93.2s ['timeline'] promoted pass=True
    v3-11 card-worthy timeline                 complete  146.3s ['timeline'] fence    pass=True
    v3-12 card-worthy timeline                 complete   28.1s ['timeline'] fence    pass=True
    v3-13 card-worthy facts                    complete   24.0s ['facts'] fence    pass=True
    v3-14 card-worthy facts                    complete  129.2s ['facts'] promoted pass=True
    v3-15 card-worthy facts                    complete  107.2s ['facts'] promoted pass=True
    v3-16 card-worthy mixed                    complete  181.5s ['comparison'] promoted pass=True
    v3-17 card-worthy mixed                    complete  171.3s ['checklist', 'timeline'] fence    pass=True
    v3-18 card-worthy mixed                    complete   22.0s ['checklist'] fence    pass=True
    v3-19 plain       none                     complete  166.3s ['facts'] fence    pass=False
    v3-20 plain       none                     complete  142.3s [] -        pass=True
    v3-21 plain       none                     complete  141.3s [] -        pass=True
```

## HELD-OUT v3 (default config, polltx, boot 95675db4)

`tests/fixtures/cards_held_out_v3.json`, sha256
`5cce5bb75b02b3c7815c5e7ecc48212e51ffc5168af17d099f95fa51985a6ebb`, frozen in
e64cc4f50 before any run. Card code: card_promotion sha256 `9f1177f9...`
(15b327e92), no card schema on ordinary chat. max_tokens 700, real HTTP
turns through the assistant, one run per model. `fence` = the model's own
valid block; `promoted` = built from markdown.

Everyday 9B (14/18 cards, 0/18 spurious):
```
    v3-01 card-worthy checklist                complete   94.3s ['checklist'] promoted pass=True
    v3-02 card-worthy checklist                complete   88.1s ['checklist'] promoted pass=True
    v3-03 card-worthy checklist                complete   96.1s ['checklist'] promoted pass=True
    v3-04 card-worthy checklist                complete   86.1s ['checklist'] promoted pass=True
    v3-05 card-worthy comparison               complete   88.1s [] -        pass=False
    v3-06 card-worthy comparison               complete   63.1s [] -        pass=False
    v3-07 card-worthy comparison               complete   84.1s ['comparison'] promoted pass=True
    v3-08 card-worthy comparison               complete   90.1s [] -        pass=False
    v3-09 card-worthy timeline                 complete   89.1s ['timeline'] promoted pass=True
    v3-10 card-worthy timeline                 complete   82.4s ['timeline'] promoted pass=True
    v3-11 card-worthy timeline                 complete   88.2s ['timeline'] promoted pass=True
    v3-12 card-worthy timeline                 complete   17.0s ['timeline'] promoted pass=True
    v3-13 card-worthy facts                    complete   24.0s ['facts'] fence    pass=True
    v3-14 card-worthy facts                    complete    7.0s ['facts'] promoted pass=True
    v3-15 card-worthy facts                    stopped    45.1s [] -        pass=False
    v3-16 card-worthy mixed                    complete   86.1s ['comparison'] promoted pass=True
    v3-17 card-worthy mixed                    complete   85.2s ['checklist'] promoted pass=True
    v3-18 card-worthy mixed                    complete    8.0s ['checklist'] promoted pass=True
    v3-19 plain       none                     complete   38.1s [] -        pass=True
    v3-20 plain       none                     complete   89.1s [] -        pass=True
    v3-21 plain       none                     complete   84.1s [] -        pass=True
    v3-22 plain       none                     complete   30.1s [] -        pass=True
    v3-23 plain       none                     complete   81.1s [] -        pass=True
    v3-24 plain       none                     complete   83.4s [] -        pass=True
    v3-25 plain       none                     complete   95.1s [] -        pass=True
    v3-26 plain       none                     complete    5.0s [] -        pass=True
    v3-27 plain       none                     complete   92.1s [] -        pass=True
    v3-28 near-miss   list-in-prose            complete   20.0s [] -        pass=True
    v3-29 near-miss   list-in-prose            complete   23.0s [] -        pass=True
    v3-30 near-miss   short-list               complete    6.0s [] -        pass=True
    v3-31 near-miss   short-list               complete   26.0s [] -        pass=True
    v3-32 near-miss   code-only                complete    3.0s [] -        pass=True
    v3-33 near-miss   code-only                complete   22.0s [] -        pass=True
    v3-34 near-miss   table-noun               complete   91.1s [] -        pass=True
    v3-35 near-miss   explanation-with-bullets complete   25.0s [] -        pass=True
    v3-36 near-miss   history                  complete   44.1s [] -        pass=True
```
Misses: v3-05, 06, 08 are comparisons written as headed sections (fixed by
rule 3 after the run); v3-15 is the repair TemplateError.

Quality 27B (15/18, 0/18):
```
    v3-01 card-worthy checklist                complete  165.3s ['checklist'] promoted pass=True
    v3-02 card-worthy checklist                complete  167.3s ['checklist'] promoted pass=True
    v3-03 card-worthy checklist                complete  216.9s ['checklist'] promoted pass=True
    v3-04 card-worthy checklist                complete  435.6s ['checklist'] promoted pass=True
    v3-05 card-worthy comparison               complete  170.3s [] -        pass=False
    v3-06 card-worthy comparison               complete  170.3s [] -        pass=False
    v3-07 card-worthy comparison               complete  177.3s ['comparison'] promoted pass=True
    v3-08 card-worthy comparison               complete  161.3s [] -        pass=False
    v3-09 card-worthy timeline                 complete  171.3s ['timeline'] promoted pass=True
    v3-10 card-worthy timeline                 complete  163.3s ['timeline'] promoted pass=True
    v3-11 card-worthy timeline                 complete  206.1s ['timeline'] promoted pass=True
    v3-12 card-worthy timeline                 complete  139.2s ['timeline'] promoted pass=True
    v3-13 card-worthy facts                    complete   57.1s ['facts'] fence    pass=True
    v3-14 card-worthy facts                    complete   14.0s ['facts'] promoted pass=True
    v3-15 card-worthy facts                    complete  122.2s ['facts'] fence    pass=True
    v3-16 card-worthy mixed                    complete  169.3s ['comparison'] promoted pass=True
    v3-17 card-worthy mixed                    complete  166.3s ['checklist'] promoted pass=True
    v3-18 card-worthy mixed                    complete   18.0s ['checklist'] promoted pass=True
    v3-19 plain       none                     complete   58.1s [] -        pass=True
    v3-20 plain       none                     complete  116.2s [] -        pass=True
    v3-21 plain       none                     complete  146.2s [] -        pass=True
    v3-22 plain       none                     complete   60.1s [] -        pass=True
    v3-23 plain       none                     complete   77.1s [] -        pass=True
    v3-24 plain       none                     complete  111.2s [] -        pass=True
    v3-25 plain       none                     complete  156.3s [] -        pass=True
    v3-26 plain       none                     complete   12.0s [] -        pass=True
    v3-27 plain       none                     complete  151.2s [] -        pass=True
    v3-28 near-miss   list-in-prose            complete   40.1s [] -        pass=True
    v3-29 near-miss   list-in-prose            complete   49.1s [] -        pass=True
    v3-30 near-miss   short-list               complete   12.0s [] -        pass=True
    v3-31 near-miss   short-list               complete   35.1s [] -        pass=True
    v3-32 near-miss   code-only                complete   38.1s [] -        pass=True
    v3-33 near-miss   code-only                complete    7.0s [] -        pass=True
    v3-34 near-miss   table-noun               complete  121.2s [] -        pass=True
    v3-35 near-miss   explanation-with-bullets complete   45.1s [] -        pass=True
    v3-36 near-miss   history                  complete   65.1s [] -        pass=True
```

Compact 4B (17/18, 0/18):
```
    v3-01 card-worthy checklist                complete   73.1s ['checklist'] promoted pass=True
    v3-02 card-worthy checklist                complete   76.1s ['checklist'] promoted pass=True
    v3-03 card-worthy checklist                complete   56.1s ['checklist'] promoted pass=True
    v3-04 card-worthy checklist                complete   44.1s ['checklist'] promoted pass=True
    v3-05 card-worthy comparison               complete   72.1s ['comparison'] promoted pass=True
    v3-06 card-worthy comparison               complete   45.1s [] -        pass=False
    v3-07 card-worthy comparison               complete   49.1s ['comparison'] promoted pass=True
    v3-08 card-worthy comparison               complete   76.1s ['comparison'] promoted pass=True
    v3-09 card-worthy timeline                 complete   50.1s ['timeline'] promoted pass=True
    v3-10 card-worthy timeline                 complete   72.1s ['timeline'] promoted pass=True
    v3-11 card-worthy timeline                 complete   74.1s ['timeline'] promoted pass=True
    v3-12 card-worthy timeline                 complete   27.1s ['timeline'] promoted pass=True
    v3-13 card-worthy facts                    complete   12.0s ['facts'] promoted pass=True
    v3-14 card-worthy facts                    complete    3.0s ['facts'] promoted pass=True
    v3-15 card-worthy facts                    complete   18.0s ['facts'] promoted pass=True
    v3-16 card-worthy mixed                    complete   74.1s ['comparison'] promoted pass=True
    v3-17 card-worthy mixed                    complete   74.1s ['checklist'] promoted pass=True
    v3-18 card-worthy mixed                    complete    3.0s ['checklist'] promoted pass=True
    v3-19 plain       none                     complete   21.0s [] -        pass=True
    v3-20 plain       none                     complete   23.0s [] -        pass=True
    v3-21 plain       none                     complete   35.1s [] -        pass=True
    v3-22 plain       none                     complete   15.0s [] -        pass=True
    v3-23 plain       none                     complete   30.1s [] -        pass=True
    v3-24 plain       none                     complete   20.0s [] -        pass=True
    v3-25 plain       none                     complete   46.1s [] -        pass=True
    v3-26 plain       none                     complete    3.0s [] -        pass=True
    v3-27 plain       none                     complete   26.0s [] -        pass=True
    v3-28 near-miss   list-in-prose            complete   18.0s [] -        pass=True
    v3-29 near-miss   list-in-prose            complete   24.0s [] -        pass=True
    v3-30 near-miss   short-list               complete    3.0s [] -        pass=True
    v3-31 near-miss   short-list               complete   11.0s [] -        pass=True
    v3-32 near-miss   code-only                complete    2.0s [] -        pass=True
    v3-33 near-miss   code-only                complete    2.0s [] -        pass=True
    v3-34 near-miss   table-noun               complete   42.1s [] -        pass=True
    v3-35 near-miss   explanation-with-bullets complete    8.0s [] -        pass=True
    v3-36 near-miss   history                  complete   17.0s [] -        pass=True
```

Old Everyday 2B (18/18, 1/18 spurious):
```
    v3-01 card-worthy checklist                complete   35.3s ['checklist'] promoted pass=True
    v3-02 card-worthy checklist                complete   37.1s ['checklist'] promoted pass=True
    v3-03 card-worthy checklist                complete   20.0s ['checklist'] promoted pass=True
    v3-04 card-worthy checklist                complete   31.1s ['checklist'] promoted pass=True
    v3-05 card-worthy comparison               complete   39.1s ['comparison'] promoted pass=True
    v3-06 card-worthy comparison               complete   37.1s ['comparison'] promoted pass=True
    v3-07 card-worthy comparison               complete   18.0s ['comparison'] promoted pass=True
    v3-08 card-worthy comparison               complete   40.1s ['comparison'] promoted pass=True
    v3-09 card-worthy timeline                 complete   29.1s ['timeline'] promoted pass=True
    v3-10 card-worthy timeline                 complete   42.1s ['timeline'] promoted pass=True
    v3-11 card-worthy timeline                 complete   37.3s ['timeline'] promoted pass=True
    v3-12 card-worthy timeline                 complete    5.0s ['timeline'] promoted pass=True
    v3-13 card-worthy facts                    complete   37.1s ['facts'] promoted pass=True
    v3-14 card-worthy facts                    complete    2.0s ['facts'] promoted pass=True
    v3-15 card-worthy facts                    complete   10.0s ['facts'] promoted pass=True
    v3-16 card-worthy mixed                    complete   26.0s ['comparison'] promoted pass=True
    v3-17 card-worthy mixed                    complete   35.1s ['checklist'] promoted pass=True
    v3-18 card-worthy mixed                    complete    7.0s ['checklist'] promoted pass=True
    v3-19 plain       none                     complete   15.0s [] -        pass=True
    v3-20 plain       none                     complete   22.0s [] -        pass=True
    v3-21 plain       none                     complete   29.1s [] -        pass=True
    v3-22 plain       none                     complete   15.0s [] -        pass=True
    v3-23 plain       none                     complete   38.1s [] -        pass=True
    v3-24 plain       none                     complete   18.0s [] -        pass=True
    v3-25 plain       none                     complete   20.0s [] -        pass=True
    v3-26 plain       none                     complete    2.0s [] -        pass=True
    v3-27 plain       none                     complete   22.1s [] -        pass=True
    v3-28 near-miss   list-in-prose            complete   13.2s [] -        pass=True
    v3-29 near-miss   list-in-prose            complete   16.0s ['timeline'] promoted pass=False
    v3-30 near-miss   short-list               complete    5.0s [] -        pass=True
    v3-31 near-miss   short-list               complete    5.0s [] -        pass=True
    v3-32 near-miss   code-only                complete    6.0s [] -        pass=True
    v3-33 near-miss   code-only                complete   19.0s [] -        pass=True
    v3-34 near-miss   table-noun               complete   29.0s [] -        pass=True
    v3-35 near-miss   explanation-with-bullets complete   12.0s [] -        pass=True
    v3-36 near-miss   history                  complete   36.1s [] -        pass=True
```
v3-29 "Describe the phases of the moon without using a list." became a
timeline from a bold-labelled numbered list (fixed by rule 4 after the run).

## HELD-OUT v2 (spent, failed on both pairs)

`tests/fixtures/cards_held_out_v2.json`: 36 prompts (18 card-worthy, 18
plain or near-miss), run with the v1 rules plus a cue-word compact schema on
every chat turn. The runner needed an `expect` field; it was added from
`category` after the first chunks crashed (prompt texts, categories and kinds
are byte-identical to the committed file, sha256 `04d6fd79...`; as-run file
sha256 `c1cf3594...`, commit `4c272ddeb`). Raw results:
`artifacts/held_out_v2_qwen3.8-2b-4bit.json`,
`artifacts/held_out_v2_qwen3.8-27b-4bit.json`.

- 2B: 14 of 36 recorded (early chunks crashed, the rerun was cut off by the
  M2 outage). 2 of 2 recorded card-worthy prompts got a card; **3 of 12
  recorded non-card prompts got a card** (v2-25 and v2-27 `comparison`, v2-35
  `checklist`): FAIL.
- 27B: 36 of 36 recorded. **9 of 18 cards** (threshold 15) and **4 of 18
  spurious** (v2-20, v2-21, v2-28, v2-36 `checklist`): FAIL. 14 turns ended
  `stopped` after 110-217 s. The runner sent its heartbeat from the polling
  loop and the server cancels a turn after 20 s without one, so those misses
  are partly a harness defect. The v3 runner sends heartbeats from a
  separate thread and records the slowest one.

```
qwen3.8-2b-4bit
    v2-01 card-worthy checklist                complete   36.3s ['checklist'] pass=True
    v2-02 card-worthy checklist                complete   40.1s ['checklist'] pass=True
    v2-25 plain       none                     complete   37.4s ['comparison'] pass=False
    v2-26 plain       none                     complete   37.1s [] pass=True
    v2-27 plain       none                     complete   25.1s ['comparison'] pass=False
    v2-28 near-miss   list-in-prose            complete    4.0s [] pass=True
    v2-29 near-miss   list-in-prose            complete    5.0s [] pass=True
    v2-30 near-miss   list-in-prose            complete    5.0s [] pass=True
    v2-31 near-miss   short-list               complete   13.3s [] pass=True
    v2-32 near-miss   short-list               complete    5.0s [] pass=True
    v2-33 near-miss   code-only                complete    3.0s [] pass=True
    v2-34 near-miss   code-only                complete    3.0s [] pass=True
    v2-35 near-miss   explanation-with-bullets complete   41.1s ['checklist'] pass=False
    v2-36 near-miss   explanation-with-bullets complete   13.1s [] pass=True
qwen3.8-27b-4bit
    v2-01 card-worthy checklist                stopped   174.6s [] pass=False
    v2-02 card-worthy checklist                stopped   181.6s [] pass=False
    v2-03 card-worthy checklist                complete   60.2s ['checklist'] pass=True
    v2-04 card-worthy comparison               stopped   217.3s [] pass=False
    v2-05 card-worthy comparison               stopped   152.6s [] pass=False
    v2-06 card-worthy comparison               stopped   109.4s [] pass=False
    v2-07 card-worthy timeline                 complete  116.4s ['timeline'] pass=True
    v2-08 card-worthy timeline                 complete   49.2s ['timeline'] pass=True
    v2-09 card-worthy timeline                 complete   34.1s ['timeline'] pass=True
    v2-10 card-worthy facts                    complete   65.2s ['facts'] pass=True
    v2-11 card-worthy facts                    complete   29.1s ['checklist'] pass=True
    v2-12 card-worthy facts                    complete   72.2s ['facts'] pass=True
    v2-13 card-worthy mixed                    complete   21.1s [] pass=False
    v2-14 card-worthy mixed                    stopped   156.5s [] pass=False
    v2-15 card-worthy mixed                    complete   38.1s [] pass=False
    v2-16 card-worthy checklist                complete   54.2s ['checklist'] pass=True
    v2-17 card-worthy comparison               stopped   187.6s [] pass=False
    v2-18 card-worthy timeline                 complete   58.2s ['timeline'] pass=True
    v2-19 plain       none                     stopped   187.6s [] pass=True
    v2-20 plain       none                     complete   94.3s ['checklist'] pass=False
    v2-21 plain       none                     complete   91.3s ['checklist'] pass=False
    v2-22 plain       none                     stopped   169.5s [] pass=True
    v2-23 plain       none                     complete   40.1s [] pass=True
    v2-24 plain       none                     complete  169.6s [] pass=True
    v2-25 plain       none                     stopped   131.6s [] pass=True
    v2-26 plain       none                     stopped   184.6s [] pass=True
    v2-27 plain       none                     stopped   187.6s [] pass=True
    v2-28 near-miss   list-in-prose            complete   47.2s ['checklist'] pass=False
    v2-29 near-miss   list-in-prose            complete   23.1s [] pass=True
    v2-30 near-miss   list-in-prose            complete   25.1s [] pass=True
    v2-31 near-miss   short-list               complete   25.1s [] pass=True
    v2-32 near-miss   short-list               complete   39.1s [] pass=True
    v2-33 near-miss   code-only                complete   33.1s [] pass=True
    v2-34 near-miss   code-only                complete   26.1s [] pass=True
    v2-35 near-miss   explanation-with-bullets stopped   173.5s [] pass=True
    v2-36 near-miss   explanation-with-bullets complete   97.3s ['checklist'] pass=False
```

## HELD-OUT v1 (spent), v0.7.6 wheel 0.32.3.dev202609291615+06711ad

`tests/fixtures/cards_held_out.json`, sha256
`0df8e489e90de6c3552115e29578bb426f589bccbaec690b46d95af62513775b` (with the
embedded `sha256` field), 24 prompts, max_tokens 700, real HTTP turns.
Raw results: `artifacts/held_out_qwen3.8-2b-4bit.json`,
`artifacts/held_out_qwen3.8-27b-4bit.json`. Wall times were recorded while
other agents' jobs shared the GPU and are not latency evidence.

### qwen3.8-2b-4bit

```
=== qwen3.8-2b-4bit: 24/24 prompts recorded ===
  card-worthy 12/12 (valid: 10)
  plain       6/6 (spurious: 1)
  near-miss   6/6 (spurious: 0)
  threshold >= 10/12: True
  threshold 0 spurious on plain/near-miss: False
    ho-01 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-02 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-03 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-04 card-worthy  comparison     expect=card  components=['comparison'] pass=True
    ho-05 card-worthy  comparison     expect=card  components=['comparison'] pass=True
    ho-06 card-worthy  comparison     expect=card  components=['comparison'] pass=True
    ho-07 card-worthy  timeline       expect=card  components=[] pass=False
    ho-08 card-worthy  timeline       expect=card  components=[] pass=False
    ho-09 card-worthy  timeline       expect=card  components=['comparison'] pass=True
    ho-10 card-worthy  facts          expect=card  components=['facts'] pass=True
    ho-11 card-worthy  facts          expect=card  components=['facts'] pass=True
    ho-12 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-13 plain        none           expect=none  components=[] pass=True
    ho-14 plain        none           expect=none  components=[] pass=True
    ho-15 plain        none           expect=none  components=[] pass=True
    ho-16 plain        none           expect=none  components=['comparison'] pass=False
    ho-17 plain        none           expect=none  components=[] pass=True
    ho-18 plain        none           expect=none  components=[] pass=True
    ho-19 near-miss    short-list     expect=none  components=[] pass=True
    ho-20 near-miss    short-list     expect=none  components=[] pass=True
    ho-21 near-miss  list-in-prose  expect=none  components=[] pass=True
    ho-22 near-miss  list-in-prose  expect=none  components=[] pass=True
    ho-23 near-miss  code-only      expect=none  components=[] pass=True
    ho-24 near-miss  code-only      expect=none  components=[] pass=True
```

### qwen3.8-27b-4bit

```
=== qwen3.8-27b-4bit: 24/24 prompts recorded ===
  card-worthy 12/12 (valid: 7)
  plain       6/6 (spurious: 4)
  near-miss   6/6 (spurious: 2)
  threshold >= 10/12: False
  threshold 0 spurious on plain/near-miss: False
    ho-01 card-worthy  checklist      expect=card  components=[] pass=False
    ho-02 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-03 card-worthy  checklist      expect=card  components=[] pass=False
    ho-04 card-worthy  comparison     expect=card  components=[] pass=False
    ho-05 card-worthy  comparison     expect=card  components=[] pass=False
    ho-06 card-worthy  comparison     expect=card  components=[] pass=False
    ho-07 card-worthy  timeline       expect=card  components=['timeline'] pass=True
    ho-08 card-worthy  timeline       expect=card  components=['timeline'] pass=True
    ho-09 card-worthy  timeline       expect=card  components=['timeline'] pass=True
    ho-10 card-worthy  facts          expect=card  components=['facts'] pass=True
    ho-11 card-worthy  facts          expect=card  components=['facts'] pass=True
    ho-12 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-13 plain        none           expect=none  components=['checklist'] pass=False
    ho-14 plain        none           expect=none  components=[] pass=True
    ho-15 plain        none           expect=none  components=['checklist'] pass=False
    ho-16 plain        none           expect=none  components=['checklist'] pass=False
    ho-17 plain        none           expect=none  components=[] pass=True
    ho-18 plain        none           expect=none  components=[] pass=True
    ho-19 near-miss    short-list     expect=none  components=['checklist'] pass=False
    ho-20 near-miss    short-list     expect=none  components=[] pass=True
    ho-21 near-miss  list-in-prose  expect=none  components=['checklist'] pass=False
    ho-22 near-miss  list-in-prose  expect=none  components=[] pass=True
    ho-23 near-miss  code-only      expect=none  components=[] pass=True
    ho-24 near-miss  code-only      expect=none  components=[] pass=True
```

The 27B cards were its own fenced JSON (checklists on plain prompts, none on
the comparison prompts), not promotion.

## Finite-logits probe (v0.7.6 wheel)

`probe_finite_logits.py`, raw output `artifacts/probe_finite_logits.json`:

```
{
  "wheel": "0.32.3.dev202609291615+06711ad",
  "model": "qwen3.8-2b-4bit",
  "max_tokens": 700,
  "elapsed_s": 14.28,
  "completion_tokens": 0,
  "has_assistant_ui_fence": false,
  "components": ["checklist"],
  "non_finite_hits": 0,
  "content_preview": "# Weekend Camping Packing Checklist\n\n## \ud83c\udf32 Shelter & Sleep\n- [ ] Tent (or tarp)\n- [ ] Sleeping bag rated for expected temperature\n- [ ] Sleeping pad or air mattress\n- [ ] Tent stakes and guylines\n- [ ]"
}
```

The 2B wrote a markdown task list without a fence; promotion built the
checklist. No non-finite logits in 700 tokens.

## What cannot be promoted

- Charts, forms, decisions and facts with sources: markdown cannot carry
  them; the full schema is sent when the user names them and only the model
  can produce them.
- A request the rules do not recognise ("what should I bring camping?")
  gets prose even if the reply is a list.
- A comparison written as loose bullets, without a table or headed
  sections, gets no card.
- Timelines need three or more dated or labelled entries.
- The parser never invents data; cells are the reply's own text, truncated
  to the validator's bounds.

## Files

- `run_suite.py`: held-out runner (v2, v3, v4; `--tag` for config runs),
  per-prompt checkpoint with card types, titles, fence or promotion, reply
  text, turn events, kernel and boot id.
- `run_v3.sh`: the v3 driver used on the M2.
- `run_latency.py`: first-text probe, 30 warm turns after 3 warm-ups.
- `run_held_out.py`, `probe_finite_logits.py`: v1 runner and probe.
- `artifacts/`: raw results (v1, v2, v3 per model, polltx latency attempt,
  stock baseline), `mc_score.py` (v1 scorer).
