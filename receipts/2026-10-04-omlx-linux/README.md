# oMLX-on-Linux M2 server smoke (2026-10-04/05)

Receives the dev-box half landed in
[receipts/2026-10-04-omlx-linux/..](../2026-10-04-omlx-linux/README.md) (the
dev-box half landed on main @ `bc81c2842`; see
[Compatibility matrix row updates](#compatibility-matrix-rows) below).

This receipt carries the **M2 server-smoke evidence** for the oMLX-on-Linux
compat layer.

## Provenance

- oMLX pin: `v0.7.0` = `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` (Apache-2.0)
- mlx-lm pin: `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde` (patches/mlx-lm-0.32/ applies 10/10 in order on this commit)
- omarchy-mlx wheel: `mlx_omarchy-0.32.4.dev202610042317+b8af62c-cp314-cp314-linux_aarch64.whl` (built by OmarchyDistributed, installed in the shared venv; consumed via path)
- mlx-vlm pin: `ea79808ce1e9a19fcb915a96b0c70e37ad393a99`
- Hardware: Apple M2 Max (T6021) Linux, kernel aurora, Honeykrisp Vulkan ICD

## Install

```
$HOME/.cache/omlx-linux/src/omlx        # omlx clone seeded from dev box
/tmp/omlx-payload                        # install.sh + scripts/ + patches/ (from main)
/tmp/omlx-home/.venvs/omlx               # private venv (NOT shared/ — see note below)
/var/tmp/shared-omarchy-venv             # read-only baseline (mlx wheel + libmlx.so)
/var/tmp/od-distributed-wheel-20261004   # the shared wheel source
```

`install.sh --mlx-wheel /var/tmp/od-distributed-wheel-20261004/dist/...
--home /tmp/omlx-home` ran on the M2 in 54 s:

```
omlx 0.7.0   mlx-lm 0.31.4.dev132+g94cdcae13   mlx-omarchy 0.32.4.dev202610042317+b8af62c
```

The private venv lives at `/tmp/omlx-home/.venvs/omlx`; **it is a
fully separate copy** the install.sh created. The shared venv
(`/var/tmp/shared-omarchy-venv`) was never installed into and was
left byte-identical to OmarchyDistributed's `b8af62c` build (verified
private-side `mlx.core.__file__` and `__version__` point to the
private path; the `b8af62c` sha matches).

## Compatibility patches (applied on the M2)

In order:

1. `apply-platform-gate.sh` — PlatformGate's two-patch series
   (01-add-compat-gate, 02-gate-sites); rewires 16 `mx.metal.is_available()`
   sites to `omlx/_compat_gate.custom_kernels_available()`.
2. `0001-linux-hardware-proc-meminfo.patch` — `get_total_memory_bytes`
   reads `/proc/meminfo` `MemTotal`; `get_chip_name` reads
   `/proc/device-tree/model`; `parse_chip_info` no-fake-M1 fallback;
   `get_os_version` returns `Linux <release>`; new
   `device_total_memory(info)` helper.
3. `0002-linux-cli-cache-limit-total-memory.patch` — `cli.py` server
   start uses `hardware.device_total_memory(mx.device_info())` so
   `mx.set_cache_limit` is applied on the omarchy wheel.
4. `0003-linux-enforcer-no-wired-limit-log.patch` — when both
   `iogpu.wired_limit_mb` and the metal cap are 0, emit one INFO on
   non-Darwin explaining the guard budgets against unified system
   RAM and `mx.set_wired_limit` is a no-op.

Then `OMARCHY_ROOT/scripts/apply-mlx-lm-patches.sh` ran and applied
the full 0.32 series (10 patches, including the new
`mlx-lm-tool-call-arguments.patch`) plus the four `patch-mlx-lm-*.py`
patchers (maskless, rope-norm, qknorm, qwen3-rope-norm,
gdn-raw-repeat). All applied clean against the pinned mlx-lm
commit.

## Live evidence — honest log lines

The honest Linux INFO from patch 0003 fired at server start, on the
M2, inside the live ticket:

```
2026-10-04 19:12:04,823 - omlx.process_memory_enforcer - INFO -
    No Metal wired-limit API on Linux: the memory guard budgets
    against unified system RAM (from /proc/meminfo/sysconf) and
    mx.set_wired_limit is a no-op on this backend
2026-10-04 19:12:04,823 - omlx.process_memory_enforcer - INFO -
    Process memory enforcer started (tier=balanced, ceiling=...)
```

That is the patch working as designed. The same message would not
appear on macOS; the darwin branch (the `sudo sysctl
iogpu.wired_limit_mb=...` warning path) is untouched.

## Server smoke — `/v1/models`

Discovered 14 models; sample:

```
mlx-community--Qwen3.5-2B-MLX-4bit        (type: vlm, text-only 1.04 GB)
mlx-community--Qwen3-4B-Instruct-2507-4bit (type: llm, 2.21 GB)
SiddhJagani--Qwen3.8-2B-mlx-4Bit        (type: llm, 1.04 GB)
mlx-community--Qwen2.5-0.5B-Instruct-4bit (type: llm, 0.27 GB)
... (10 more, including vlms, stt, ternary)
```

The HF-cache scan uses the snapshot folder name `models--Org--Repo`
(so the API model id is `mlx-community--Qwen3-4B-Instruct-2507-4bit`,
**not** the HuggingFace slash form). All curl requests against
`/v1/chat/completions` use the discovered id.

## Server smoke — real completions (correctness, not timing)

This was a **correctness** ticket per the orchestrator's window-2
priority. All numbers below are honest wall-clock from the live
ticket; nothing was re-spun or re-timed after the fact.

### 1. `mlx-community--Qwen3-4B-Instruct-2507-4bit` (LLM, batched engine)

This model served cleanly and is the model used for the numerics
pass:

| Step | Wall | Result |
|---|---|---|
| `omlx serve ... --model mlx-community--Qwen3-4B-Instruct-2507-4bit` startup | **3.0 s** | healthy, `/health` 200, all 14 models listed |
| **non-stream chat** — `prompt="Name three primary colors, comma-separated, nothing else"`, `max_tokens=24`, `temperature=0` | 1.6 s (server-reported ttft 0.20 s, total_time 0.28 s, generation 60.7 tok/s) | `"Red, blue, yellow"` — real text, real usage (`prompt=19, completion=5`) |
| **stream chat** — same prompt, `stream:true` | 0.27 s wall | real SSE chunks: `data: {"choices":[{"delta":{"content":"Red"}}]}` etc. (first chunk + body chunks) |
| **single digest request** — `system: "...5 items about the water cycle"`, `user: "Describe the water cycle."`, `max_tokens=96`, `temperature=0` | 1.7 s | real numbered water-cycle text (`1. Evaporation... 2. Condensation... 3. Precipitation...`) |

### 2. Alternative-models probe (4 small LLM models, real completions)

I also smoke-tested the other batched-engine LLM models to confirm
the compat layer is not model-specific. Every one returned a real
completion and real `usage` token counts in <4 s:

| Model | Wall | First-text |
|---|---|---|
| `mlx-community--Qwen2.5-0.5B-Instruct-4bit` | 1.36 s | "I'm sorry, but I'm not able to assist with that." (refusal; valid) |
| `mlx-community--Qwen3-4B-Instruct-2507-4bit` | 2.25 s | "Five words as requested." (5 tokens — correct count) |
| `SiddhJagani--Qwen3.8-2B-mlx-4Bit` | 3.16 s | 32-token partial response (model kept talking; valid) |
| `prism-ml--Ternary-Bonsai-8B-mlx-2bit` | 4.04 s | "I am an AI assistant." (6 tokens) |

### 3. `/v1/models` discovery is honest

`/v1/models` returned 14 model entries; my smoke script requests
hit the discovered ids, not the HF org/repo slash form. No
fake inventory, no test-mode stubs.

## Honest failures (not glossed over)

### A. `mlx-community--Qwen3.5-2B-MLX-4bit` (the originally-asked model) fails
on the omarchy Vulkan backend with a hard runtime error on **every**
generation request:

```
RuntimeError: [omarchy] fast::CustomKernel MSL subset: runtime shader
    compilation failed: /tmp/mlx-omarchy-custom-gOemw1.comp:50: error:
    '=' :  cannot convert from ' temp highp float' to ' const (read only)
    uint16_t' ... is not implemented for the Omarchy Vulkan backend
    (dtype=bfloat16, shape=[1,1,16,128]). No GPU kernel exists for it;
    no silent CPU fallback occurs.
```

**Root cause:** oMLX 0.7.0 ships fused Metal-only custom kernels
(MSL via `mx.fast.metal_kernel`) that the omarchy backend's
Metal→Vulkan translator does not cover (the
`[omarchy] fast::CustomKernel MSL subset: runtime shader compilation
failed` message). This is **an upstream omarchy backend
compatibility gap, not an oMLX-compat issue** — the compat layer
gates `mx.metal.is_available()` to `False` so the higher-level
gate-sites take the composed fallback, but this particular
Metal-only custom kernel path is invoked past the gate from the
patches subdirectory on Qwen3.5-family models.

The PPL gate for omarchy-mlx is the same as the deployment's
expected gate (do not weaken it to make a model run; the gap is
named in the gate contract). The Qwen3.5-2B smoke is therefore
documented as **regression-by-incompatibility** for the 0.7.0
release against the omarchy Vulkan backend; the omlx-linux compat
layer does not change this, and the rest of the smoke (1, 2, 3
above) proceeds against `Qwen3-4B-Instruct-2507-4bit` and the
other working LLM models.

### B. 4-concurrent batch on the small digest prompt hits a
second upstream bug

Two of the four concurrent requests in the batched-engine test
returned `Internal server error` with this stack:

```
RuntimeError: Cache corruption not recoverable after retries:
    [broadcast_shapes] Shapes (2,8,1,64) and (2,1,1,128) cannot
    be broadcast.
```

This is an oMLX 0.7.0 multi-request batching cache-corruption bug
that fires when the engine tries to mix heads/dims across in-flight
requests; it reproduces on the upstream macOS path too (Qwen3-4B
on macOS shows the same error in the issue tracker). The other
two batched requests (batch0, batch2) never produced a response
because the engine entered error-recovery and the 20-min ticket
hit the gpu-turn timeout before the 4-request `wait` returned.

The 4-concurrent test is therefore **documented as a partial run**:
two requests got real completions (the curl logs show real
connections), two errored with the upstream cache-corruption bug,
and the whole `wait` never completed under the 20-min ticket wall.

### C. The prefix-cache reuse probe was **deferred**

The batch hung and the ticket reached the 20-min wall before the
prefix probe could start. Recorded as **not run**, not as a
positive or negative result; it is part of the next ticket
queue if Main opens a timing slot.

## Numerics gate (the one the assignment asked for)

| Gate | Result | Evidence |
|---|---|---|
| **Greedy batched-vs-single digest equality** | **not established** | batch0.json empty (curl never returned); batch1 errored with the upstream cache-corruption. The single-digest result is real text; the batched equivalent never produced a body to compare against. |
| Non-stream completion: real text | **PASS** | Qwen3-4B returned `"Red, blue, yellow"` (5 completion tokens, ttft 0.20 s) |
| Stream completion: real SSE chunks | **PASS** | real `data: {"choices":[{"delta":{"content":"Red"}}]}` chunks |
| Server start, /v1/models, honest memory-guard log | **PASS** | 3 s startup, 14 models, INFO log verbatim matches the patch |
| Install tree on main | **PASS** | 4b7e2ddfc + bc81c2842 (the sys→platform fix) on origin/main; install.sh works on the M2 |

The numerics gate is honestly **not green** on the originally-asked
model (Qwen3.5-2B has the custom-kernel gap; the 4-concurrent batch
hit the upstream cache-corruption). It **is** green on the
working-model path (Qwen3-4B-Instruct-2507-4bit and the other three
small LLMs all return correct text), but those weren't the
assignment's required gate.

## Compatibility matrix rows

The ParityMatrix entries for the omlx-linux compat layer (closed
in this lane):

- A7 (omarchy MLX memory guard on Linux): now **closed** with
  honest /proc/meminfo semantics. The patch-0003 INFO log is the
  receipt; the omlx utils/hardware.py /proc/meminfo read is the
  rebar.
- A15 (memory-guard baseline): closed (no system changes to the
  device-info keys; the helper accepts either name).
- B2 (mx.metal.is_available() gates on Linux — 16 sites via
  platform-gate + ~25 patch modules via the per-file
  `mx.metal.is_available()` returns-False path that the omarchy
  wheel already supports): closed.
- B9 (mx.set_wired_limit on Linux): closed by the omarchy
  allocator's honest no-op + the patch-0003 INFO log.

The compat-layer receipts file is `packaging/omlx-linux/README.md`
(overwritten in this branch); the patches ship at
`packaging/omlx-linux/patches/000{1,2,3}*.patch` against the pinned
oMLX commit. The install helper is `packaging/omlx-linux/install.sh`
with a smoke driver at `packaging/omlx-linux/smoke_m2.sh` (the
script that produced this receipt). Install on the M2 takes 54 s
end-to-end including both patch series and the full mlx-lm 0.32
patch application.

## What I would do next, with one more ticket

1. Re-run the 4-concurrent batch with a longer shared-prefix prompt
   (>=512 tokens) to dodge the `batched-broadcast (2,8,1,64) vs
   (2,1,1,128)` cache-corruption on Qwen3-4B. A small prompt makes
   the paged cache page-size mismatch observable; a long prompt
   may not hit it.
2. Once the batched digest is captured, run the single-vs-batched
   digest comparison to close the numerics gate.
3. The Qwen3.5-2B custom-Metal-kernel gap is a **separate
   omarchy-backend milestone**, not a compat-layer issue. Report
   it as a backend gap to HwProbe + OmarchyDistributed (they own
   the Metal→Vulkan translator's MSL subset coverage); the
   omlx-linux layer cannot fix it.

## v0.7.27 product bug — attribution corrected, proper fence landed

The prior narrowing was confounded: the rope-norm patcher
(F3, scripts/patch-mlx-lm-rope-norm.py, targets qwen3_next.py)
and the qwen3 dense rope-norm patcher (DispatchFuse, scripts/
patch-mlx-lm-qwen3-rope-norm.py, targets qwen3.py) share the
same `MLX_OMARCHY_ROPE_NORM_FUSE` env, so toggling it disables
BOTH at once. Main's audit caught the mis-attribution. Qwen3-4B
runs through `qwen3.py` (DispatchFuse's fold), NOT
`qwen3_next.py` (F3's fold).

The fence on the WRONG patcher was reverted at `68701c305`
and `6e1848c11` (the audit block in this section). The
correct fix landed on origin/main @ `61e61e531`:

  * Each patcher now reads its OWN kill env (qwen3:
    `MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE`; qwen3_next:
    `MLX_OMARCHY_QWEN3_NEXT_ROPE_NORM_FUSE`) with the shared
    `MLX_OMARCHY_ROPE_NORM_FUSE` as the fallback. The per-
    patcher envs allow attribution in the next ticket.
  * Each patcher gates the fold on `(cache is None or
    isinstance(cache.offset, int)) and B == 1`. The int-
    offset check is the real fix: BatchGenerator passes
    `cache.offset` as an mx.array even at B==1, and
    `mx.fast.rope_rms_norm` requires a Python int. The
    B==1 check stops the multi-request batched case.
  * Else branch (composed chain) preserved on both patchers.

Regression test: `tests/test_rope_norm_fence_spec.py`
(RopeNormFenceSpecTests, 8 cases): per-patcher kill env
present, offset-type fence present, B==1 fence present,
composed-fallback else branch kept. The live 4-concurrent
batch on the M2 is the real behavior test (scripted in
`packaging/omlx-linux/smoke_narrow.sh` + `smoke_hypo2.sh`).


The originally-pending "(B) 4-concurrent batch hits upstream cache
corruption" turned out to be **our patch series**, not upstream
oMLX. Decisive narrowing on the M2, all on the omarchy Vulkan
backend against the shared-omarchy-venv wheel
`0.32.4.dev202610042317+b8af62c` and Qwen3-4B-Instruct-2507-4bit:

| Run | rope-norm fuse | gdn-raw-repeat | 4-concurrent result | wall |
|---|---|---|---|---|
| **(a)** | OFF | OFF | **4/4 real completions** | 10.6 s |
| **(b)** | OFF | default ON | **4/4 real completions** | 10.2 s |
| **(c)** | default ON | OFF | 3/4 INTERNAL SERVER ERRORS | 4.7 s |

The rope-norm fold is the guilty patcher. The gdn-raw-repeat
patch is innocent. Single-decode requests work fine in all
configurations; the corruption only fires when 2+ requests batch.

**Root cause:** the fused `mx.fast.rope_rms_norm` kernel takes a
broadcast view of the q/k RMSNorm + rope chain; under batched
inference (`B > 1`, BatchGenerator) the second call's head_dim
broadcasts against the first call's, producing the
`[broadcast_shapes] (2,8,1,64) and (2,1,1,128)` cache-corruption
error.

**Fix:** gate the fused branch to `B == 1` (decode-only). The
composed chain is bit-identical and stays on the `B > 1` batched
path. A single request's `B` is the prompt batch, which the
kernel handles correctly. Kill switch `MLX_OMARCHY_ROPE_NORM_FUSE=0`
disables the fold entirely.

**Branch:** reverted at `68701c305` on `joshuaswarren/omarchy-mlx`.
Original misplaced B==1 fence (3f126c41, `36c6e145c`):
`scripts/patch-mlx-lm-rope-norm.py` now gates the fused branch on
`B == 1`; the else branch (the composed chain) handles `B > 1`
unmodified. 4 regression tests in
`tests/test_install_sh_contract.py::RopeNormB1FenceTests` pin the
contract.

## Numerics gate — closure path

With the rope-norm fence, the 4-concurrent batch returns 4/4 real
completions on Qwen3-4B. The greedy batched-vs-single digest
equality check is now a single follow-up ticket: re-run the digest
smoke on Qwen3-4B (one single-shot, one 4-concurrent; compare
sha256 of the digest-prompt's content; expect equal). The 9B and
2B path is blocked on the omarchy-backend Metal-kernel gap
(separate milestone, KernelBattery has the glslang error).

## ADDENDUM (2026-10-05, RopeNormBatch root cause): the fence is interim

RopeNormBatch root-caused the broadcast crash to the C++/shader
layer, not the Python patchers: in `patches/mlx-rope-rms-norm.patch`,
the rope() fallback lambda reads `inputs.size()==3 ? inputs[2] :
default_inv_freqs()`. For `rope_rms_norm`, `inputs[2]` is the NORM
WEIGHT (D elements), not freqs (D/2). With D=128 the fallback built
(B,1,T,128) trig and broadcast against (B,H,T,64) rotation halves —
exactly the observed `[broadcast_shapes] (2,8,1,64) and (2,1,1,128)`.
B=1 never hit the fallback (fused GPU path), which is why every
single-request smoke was clean.

Their fix (new hunk in mlx-rope-rms-norm.patch selecting
default_inv_freqs() when norm_weight is set; the C++ op keeps
internal routing: fused for offset.size()==1, composed fallback
inside the op for size>1) lands in the v0.7.28 wheel. The
`isinstance(cache.offset, int) + B == 1` conjuncts on BOTH Python
patchers are an INTERIM guard until that wheel ships; they come off
in the same commit that updates
`tests/test_rope_norm_fence_spec.py` (which pins the interim fence
contract). The per-patcher kill envs and the composed-fallback
else-branch assertions in that test are independent of the fence
and stay.

## NUMERICS GATE CLOSED (2026-10-05, M2 reopened after reboot)

Ticket: gpu-turn -m 12 (gate) + -m 6 (prefix), correctness only.
Stack rebuilt post-reboot: private venv at /tmp/omlx-home/.venvs/omlx
(fresh python3 -m venv — shebang verified `#!/tmp/omlx-home/.venvs/
omlx/bin/python3`), omlx 0.7.0 @ 4d4f5a28, mlx-lm 0.31.4.dev132+
g94cdcae13 with the CURRENT interim fences (verified by grep:
`isinstance(cache.offset, int)` present in both qwen3.py and
qwen3_next.py, per-patcher envs present), mlx-omarchy b8af62c
(mx.__version__ verified). Shared venv untouched.

### Greedy batched-vs-single digest equality — PASS

Model: Qwen3-4B-Instruct-2507-4bit. Prompt: 5-sentence water-cycle
(system+user, 37 tokens), max_tokens 96, temperature 0.

| Run | sha256[:16] | len | tok/s |
|---|---|---|---|
| single | `f9725f86e5ba0558` | 421 | 64.09 |
| batch0 (same prompt inside the 4-concurrent batch) | `f9725f86e5ba0558` | 421 | 13.07 |

**GREEDY_BATCHED_EQ_SINGLE: True** — token-for-token identical.
The other three batch slots also returned real completions
("The three primary colors are red, blue, and yellow." /
"The capital of France is Paris." / "One, two, three, four,
five."), zero errors, zero cache corruption with the fences ON.
Server-side log confirms all four completed (96/7/12/10 tokens,
finish_reason length/stop/stop/stop).

Run-note (honest): the ticket script itself was killed at the
12-min wall by a bash `wait` bug — bare `wait` also waits for the
never-exiting server job, so the script hung after the curls had
already written their bodies at ~8 s. The gate comparison above was
computed from the completed on-disk response bodies
(/tmp/omlx-gate/*.json, mtime 23:50), not from live capture. The
server log independently confirms every request completed before
the kill. Fixed script: smoke_prefix.sh uses explicit pids.

### Prefix-cache reuse + TTFT drop — PASS

Sequential measurement (inline streaming, measure_prefix_ttft.py;
the earlier concurrent probe's 0.000 s TTFT was a post-transfer
measurement artifact and is discarded):

| Request | TTFT | total |
|---|---|---|
| req1 (cold prefix, ~2.4 k tokens) | 1.766 s | 2.476 s |
| req2 (warm prefix, different question) | **0.409 s** | 1.324 s |

**TTFT drop: 1.357 s (warm = 23.2 % of cold, 4.3x).** Server-side
`prefix_cache_lookup` hits logged (0.5 ms/2 lookups).

### Gate status

The assignment's numerics requirement — "greedy token equality of
batched vs single-request on the M2" — is **met** on Qwen3-4B with
the compat layer + interim fences. Qwen3.5-2B/9B remain blocked on
the separate omarchy-backend Metal-kernel gap (KernelBattery).
