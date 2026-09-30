# Card promotion receipt, 2026-09-30

Cards on the default pairs without the model emitting JSON: when a chat turn
ends without a valid `assistant-ui` fence, `card_promotion.extract_text`
builds at most one card from the reply's markdown. On 8a1e25843 the pairs
became Everyday = Qwen3.5-9B, Compact = Qwen3-4B-Instruct-2507, Quality =
Qwen3.8-27B (the 2B left the catalog).

**No pair is qualified by this receipt yet.** The frozen v3 suite has run on
all four chat models; the config-selection run, HELD-OUT v4 and a valid
latency measurement are still to run (the M2 is shared and was rebooting for
another workstream's kernel-module tests).

## Kernel of every row

| Rows | Kernel | Boot |
|---|---|---|
| v1, v2, finite-logits probe | polltx (7.1.13-ARCH-polltx) | several, 2026-09-29/30 |
| v3 on 2B, 9B, 27B, 4B; first latency attempt | polltx | 95675db4 |
| Stock baseline below; fenced-json v3, v4, final latency | stock 7.1.13-3-1-ARCH | 4aed18b3 and later |

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
| v3 fenced-json config (selection only) | not run yet | not run yet | not planned |
| v4 (frozen, sha256 `7f807008...`) | not run | not run | not run |
| first-text p95 <= 2.0 s, 30 warm turns | not measured validly | not measured validly | not required |
| `extension.card_format` | not set | not set | not set |

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

## Latency: the first attempt is not valid

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
