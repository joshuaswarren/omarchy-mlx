# Split-K decode attention on G13G, G13C and G14C (2026-10-23)

Decode attention for bf16, head width 128 and one query token now has a two-dispatch route that spreads the keys over blocks. Qwen3-4B 4-bit decodes 12 to 23 percent faster at a 477-token prompt. It decodes 68 to 97 percent faster at a 1888-token prompt. Each number is the median of 5 interleaved rounds on one of three Apple chips.

The route is the default from a per-part key count. The counts are 128 keys on the Apple M1 (G13G), 320 on the Apple M1 Max (G13C) and 448 on the Apple M2 Max (G14C). Shorter caches keep the fused kernel, which is faster there. A set `MLX_OMARCHY_SDPA_DECODE_SPLIT` wins everywhere.

The output is not bit-equal to the fused kernel. The route sums in float32 in a different order. At 477 prompt tokens the greedy digest equals the old one in every round on all three chips. At 1888 prompt tokens it differs, identically on all three chips, at one exact bf16 tie (section "Greedy divergence").

A 7600-token teacher-forced run on the Apple M2 Max fails one of three limits that were written down before the run. Perplexity against a float32 reference rises by 0.27 percent over the old route on that text. Three controls with other summation orders in stock ops move the number by +0.0033, +0.0002 and -0.0001 nats per token against the old route. The cause was not found. Sections "Long context" and "Controls" give the data.

The rule is the pure function `sdpa_split_min_keys(device_name)` in `device.h`. `sdpa_split_default(device_name, k_len)` returns true when the minimum is nonzero and the cache holds at least that many keys. `MLX_OMARCHY_SDPA_DECODE_SPLIT=0` restores the old route at run time. Any other non-empty value forces the route at every length it covers.

All tables are tab-separated files in `tables/`. Every number below is read from a log in this directory.

## Machines

The Apple M2 Max machine is a 14-inch model with Vulkan device Apple M2 Max (G14C B1). The Apple M1 Max machine is a 16-inch model (G13C C0). The Apple M1 machine is a 13-inch model (G13G). All three use the same Mesa build `vulkan-f144c7949b` for the MLX runs. The governor is performance. `call_settle_us` is 20 on the Apple M2 Max.

Kernel strings are named with every number. "Stock" is 7.1.12-2-20261013.18-sep-ARCH. "egpu" is 7.1.12-2-20261013.18-egpu-ARCH. The Apple M1 Max ran its first screens on the egpu kernel and its final screens on 7.1.12-2-12.6-sep-ARCH, an older stock build. A number is only compared with numbers from the same run.

## Stack

Two trees were built from the branch and unpacked from their wheels. The break-even sweeps and the first decode screens ran on commit 01e82560 (mlx_omarchy 0.32.4.dev202610230958+01e8256, libmlx prefix `031d1d1063630006`). The final speed, long-cache, numerics and divergence runs ran on commit 6943e37a (0.32.4.dev202610231240+6943e37, libmlx prefix `8533393455ca0f8c`). Both commits have the same kernels and routing. The later one adds the per-part default, its tests and the docs text.

The ledger arm of every screen is the installed wheel mlx_omarchy 0.32.4.dev202610231001+48908ca (libmlx `a9738db49ca96906`). That wheel already has the chunked vocabulary-tail default on G14C. `provenance/<chip>.txt` holds the output of `scripts/mlx_provenance.py` for the final tree on each chip. The loaded `mlx.core` extension and `libmlx.so` hash to the wheel RECORD, and the compiled version equals the distribution version. Every run on the final commit used this unpacked tree.

The model is Qwen3-4B 4-bit from mlx-community (index sha256 prefix `f7825defe5865d17`), with 36 layers and 32 query heads over 8 key-value heads of width 128.

## Break-even sweep

Files: `sweeps/sd3-*.tee`, table `tables/break-even-chain-us.tsv`. The sweep times single-query grouped-query attention at cache lengths 128 to 1024. Each point is the median of 7 repetitions. Two shapes of work run: 36 independent calls on distinct caches, and a 36-call chain where each output is the next query. The chain is the shape of a decode step.

The split route costs 79 to 90 microseconds per call up to 512 keys because it takes two dispatches. The cost reaches 89 to 113 microseconds at 1024 keys. The fused kernel walks the keys serially and costs about 0.7 microseconds per key on the Apple M1. The thresholds are the smallest measured length where the split route is faster in both shapes of work. On independent calls the Apple M1 reads 76.8 against 76.1 microseconds at 128 keys. The Apple M1 Max reads 84.0 against 78.2 at 320 keys. The Apple M2 Max reads 79.4 against 78.6 at 448 keys. The first sweep (`sweeps/sd2-*.tee`) shows the same crossover near 256 and 500 keys.

## Long cache

Files: `sweeps/sd4-*.tee`, table `tables/long-cache-chain-us.tsv`. The final commit with default routes runs against `MLX_OMARCHY_SDPA_DECODE_SPLIT=0`. That arm is the fused kernel up to 7168 keys and the composition above. Each point is a chain of 12 calls. The Apple M1 and Apple M2 Max rows ran on the stock kernel. The Apple M1 Max row ran on 7.1.12-2-12.6-sep-ARCH.

At 8192 keys the old route takes 11264, 2924 and 2993 microseconds per call on the Apple M1, M1 Max and M2 Max. The split route takes 739, 323 and 236. At 32768 keys the numbers are 44494 against 2571, 10986 against 1032 and 11480 against 708. The measured range ends at 32768 keys. The route is allowed up to 1048576 keys, and pass 2 folds the blocks serially per query head (256 blocks at 32768 keys).

## Speed

Files: `screens/*.tee`, tables `tables/speed-477-tokens.tsv` and `tables/speed-1888-tokens.tsv`, summary script `scripts/armsum.py`. The step is the ledger LLM step in `scripts/llmstep.py`. It runs greedy decode of 128 tokens from the prompt, one process per sample. Every arm runs once per round and the order rotates by one place each round. Decode prints at 0.1 tokens per second.

The final tree has two arms in one libmlx. `default` has the flags unset. `split0` sets `MLX_OMARCHY_SDPA_DECODE_SPLIT=0`. The difference between them is the attention route. On the Apple M2 Max both arms also carry the chunked vocabulary-tail kernels. The third arm, `ledger`, is the installed wheel.

At 477 prompt tokens (cache 477 to 604 keys) the change is +22.62 percent on the Apple M1 (20.5 against 16.8 tokens per second), +18.53 on the Apple M1 Max (61.4 against 51.8) and +12.43 on the Apple M2 Max (63.2 against 56.3). The digest is `8e68ea1ec2d62fcc` in every arm.

The first decode screens on commit 01e82560 compared `base` and `split` in one tree with the vocabulary-tail flags off. They gave +21.89 percent on the Apple M1 (stock), +19.54 on the Apple M1 Max (egpu kernel) and +12.80 on the Apple M2 Max (egpu kernel). Every digest equalled `8e68ea1ec2d62fcc`.

At 1888 prompt tokens (cache 1888 to 2016 keys) the change is +68.37 percent on the Apple M1 (16.5 against 9.8), +96.81 on the Apple M1 Max (55.5 against 28.2) and +78.77 on the Apple M2 Max (58.1 against 32.5). The 2000-token request rounds to 111 repeats of the prompt sentence. The default digest is `4232861db0522fd8` and the old-route digest is `8e68ea1ec2d62fcc`, identical on all three chips.

The ledger arm equals `split0` within the print resolution in all six screens. The Apple M2 Max 477-token screen read a load average of 1.36 at the end of its run, from the decoding process itself. The start gate is 1.3 and passed at 0.09.

## Numerics

### Per-op gate

`tests/omarchy/test_sdpa_split.cpp` compares the split route and the fused route with a float64 reference. It covers group sizes 1, 2, 4 and 8, strided cache views, one-hot values with uniform scores, long and large-magnitude inputs, misaligned keys, uncovered shapes, the flag unset at 300 and 1000 keys on the running device, and the part rule at its boundaries.

The maximum error of the split route against float64 must be at most the fused route's maximum error plus two half steps of the largest reference value. The mean must be at most the fused mean plus a quarter of that half-step pair. The maximum must be at most three steps. The route must take exactly two dispatches.

The binaries were built on the Apple M2 Max from the final commit. Their sha256 values are `f11e9552c6a13b0c3b4d7baa16611acf7e2934aee0857550e74b38a83f3ae11f` for the split tests and `825528a0a3dba269870f83a339b158f3ecfc55601e5fccb18d5538121a79099d` for the fused tests. On the Apple M2 Max, the Apple M1 Max and the Apple M1, `omarchy_sdpa_split_tests` passes 7 of 7 cases and 217 of 217 assertions. `omarchy_sdpa_decode_fused_tests` passes 7 of 7 cases and 100490 of 100490 assertions. It is pinned to the fused route. No case was skipped. Logs are in `run-logs/`.

### Short text against a float32 reference

Files: `numerics/*short*`, `numerics/*ref*`, table `tables/numerics-short-636-steps.tsv`. The scripts `sdpa-num.sh` and `sdpa-numref.sh` run Qwen3-4B one token at a time with teacher forcing. Four segments of 160 tokens give 636 scored steps. The cache stays under 160 keys. `MLX_OMARCHY_SDPA_DECODE_SPLIT=1` forces the route.

The reference arm uses the same quantized weights with the floating tensors cast to float32. Attention and every other layer then run in float32. That is a different arithmetic from both bf16 arms and not ground truth for attention alone.

The reference NLL is 1.56809. The old route agrees with it on 631 of 636 steps (5 flips, NLL +0.2396 percent, mean difference 0.03823). The split route agrees on 629 (7 flips, +0.2474 percent, 0.03607). Flip gaps against the reference run to 0.2052 for the old route. They run to 0.5235 for the split route.

Split against old route: top-1 equal at 628 of 636 steps, mean difference 0.04572, maximum 1.12477. The Apple M1 and the Apple M2 Max print the same figures, with 82 and 79 bit-equal steps. Old route against a repeat of itself is bit-equal at 636 of 636 steps. The Apple M1 Max capture of the non-reference arms (egpu kernel) matches the Apple M1.

The limits were written before the run. The split mean difference must be at most 1.05 times the old route's (0.03607 against 0.04014: pass). Agreement must be at least the old route's minus 2 steps (629 against 629: pass at the edge). The NLL change must be at most the old route's plus 0.01 percentage points (+0.0078: pass).

### Long context

Files: `numerics/apple-m2-max-long/` (`paired.txt`, `analysis.txt`, run logs, captures), tables `tables/numerics-long-7600-steps.tsv` and `tables/order-controls-7600-steps.tsv`. One segment of 7600 tokens runs on the Apple M2 Max. It is the 35 KB text three times, so positions above about 2500 repeat earlier text. Cache lengths run to 7600. The old route is the fused kernel up to 7168 keys and the composition above. The split route runs at every length.

The old-route and split captures and the first reference ran on two stock boots. The control arms ran on two egpu boots. The run logs carry the boot ids. Capture hashes are in the run logs.

The reference NLL is 1.39471. The old route reaches 1.39821 (7488 of 7599 top-1, 111 flips, +0.2507 percent). The split route reaches 1.40089 (7498 of 7599, 101 flips, +0.4426 percent).

The limits were written before the run. The mean difference must be at most 1.05 times the old route's (0.03510 against 0.03692: pass). Agreement must be at least the old route's minus 20 steps (7498 against 7468: pass). The NLL change must be at most the old route's plus 0.05 percentage points. That third limit fails. The split route is +0.19 percentage points above the old route.

The paired difference by step is +0.00268 nats per token for split minus old route. The 95 percent bootstrap interval over steps is +0.00055 to +0.00486. Steps are correlated, so the interval is a floor. The difference is positive in 7 of 8 position bins of 950 steps and in all three thirds of the text. The limit of 0.05 percentage points is far below the old route's own distance from the reference (0.25 percent). The failure stands.

### Controls

Four more bf16 arms differ from the old route only in how the single-query attention sums over the keys. Each takes attention from stock MLX ops in float32. Scores, softmax and P@V run in float32 and the output rounds once to bf16. No kernel is involved (`scripts/tfref-attn32*.py`). Each arm made 273564 patched calls, which is 36 layers times 7599 steps. A start-up self-test against float64 passed for each.

The plain arm sums over the keys in cache order. It equals the old route bit for bit at 7599 of 7599 steps. The fused kernel is documented as bit-identical to this composition, and the per-op test below confirms it. So the plain arm duplicates the old route by construction.

The `blk` arm runs the split-K algorithm in float32 stock ops. It uses blocks of 128 keys, a block maximum, exp, a block sum, weighted values, a global-maximum rescale and a fold. Its NLL is 1.39807, which is +0.00336 against the reference. The `perm` arm sums over a random permutation of the keys. The permutation is seeded by the cache length, so it is new at every step. Its NLL is 1.39845 (+0.00373). The `rev` arm reverses the key axis. Its NLL is 1.40152 (+0.00681).

The old route is +0.00350. The split kernel is +0.00617. Paired against the old route, `blk` reads -0.00014 (interval -0.00229 to +0.00201), `perm` +0.00024, the split kernel +0.00268 and `rev` +0.00331 (interval +0.00107 to +0.00540). `rev` minus split is +0.00063 (interval -0.00139 to +0.00245). `blk` minus split is -0.00281 (interval -0.00491 to -0.00082). `numerics/apple-m2-max-long/paired.txt` lists every paired difference with its interval. `scripts/paired-controls.py` produces it from the seven captures.

Three arms lie within 0.0004 of each other: the old route, `blk` and `perm`. The split kernel and `rev` lie 0.003 higher, within 0.0007 of each other. The five values are not a smooth spread. A summation order that differs from the old route by float32 rounding alone can move the long-context NLL by this much. One stock-op arm moves it by more than the split kernel does.

A second float32 reference with the reversed key order has the same mean NLL as the first (1.39471). Its per-step difference from the first reference is at most 3.4e-4 nats and 1.1e-5 on average (last line of `paired.txt`). So the float32 model does not depend on the order. The gap of the old route is not an artifact of sharing a summation order with the reference. Against the second reference the gaps are +0.2506, +0.4425 and +0.4880 percent for the old route, the split kernel and `rev`.

The reading rules for these arms were written before each run (`prereg.txt`). The rule for the second reference expected `rev` to become the close arm. That did not happen. The `blk` outcome also differed from the rule: it equals the old route and not the split kernel.

Excluded by data: float32 partial sums, because the kernel's scratch buffer is `float` and pass 2 folds in float32 and rounds once. Excluded as well: bf16 rounding of intermediates in the old route, which is float32 attention. The per-op test below excludes a bias.

### Per-op error against float64

Files: `numerics/bias-*`, script `scripts/sdpa_bias.py`. The test draws random bf16 queries and caches at decode shapes. It uses 32 query heads over 8 key-value heads of width 128, 300 to 7000 keys and score standard deviations of 1, 3 and 5. It runs 30 trials on the Apple M1 and 60 on the Apple M1 Max. It compares the old route, the split route (flag read live) and the float32 composition with a float64 reference.

The mean absolute error is 0.25 bf16 steps for every arm, which is the output rounding. The relative bias lies between -1.6e-5 and +1.4e-5 for both routes and agrees between them within 1e-6. The split route differs from the old route in 0.01 to 0.2 percent of the output elements, always by one bf16 step. That is about 45 flipped elements per token. The old route equals the float32 composition at 1.0000 bit-equality with a maximum difference of 0.00 steps. No bias above 0.02 steps is present. A smaller bias is outside what this test can show.

### Greedy divergence at 1888 prompt tokens

Files: `numerics/*-gdiv/`. The script `gen-div.sh` generates 128 greedy tokens from the 1888-token prompt in three arms: default, `SPLIT=0` and a float32 reference. It saves the top-1 minus top-2 log-probability gap at every step. The Apple M1 and the Apple M2 Max (stock) print identical results.

The default arm and `SPLIT=0` first differ at generated token 17. The default arm emits token 220. The `SPLIT=0` arm and the reference emit token 8162. The gap at that step is 0.0 in the default arm, an exact tie in bf16. It is 0.125 in the `SPLIT=0` arm and 0.0259 in the reference arm. The reference and the `SPLIT=0` arm agree for all 128 tokens.

The reading was written before the run. A gap of at most 0.25 nats in the old arm, with the reference choosing either arm's token, counts as a near-tie. One prompt and one continuation make one example and not a rate.

## Limits

One model, one text, prompts of 477 and 1888 tokens, one chip per part and one boot per run. The default applies to every unmeasured bf16 head-width-128 model with 1, 2, 4 or 8 query heads per key-value head on these three parts. Speed was measured to 32768 keys and numerics to 7600 keys. The float32 reference differs from both bf16 arms in every layer. The long-context numerics ran on the Apple M2 Max only. The Apple M1 Max contributes its 636-step capture, its decode screens and its per-op test.

## Timing and thermal procedure

Every timed run was one queue job with no other GPU or ANE job running. The load average was read at the start. The scripts abort above a load of 1.3 at their start and print the load at every sample. No temperature reading was taken. Samples run as one process each, the arms interleave and the order rotates. The numerics, divergence and doctest runs are untimed.

## Commands

Break-even: `sdpa_bench3.sh OUT 7 timed` with `sdpa_bench_fine.py`. First sweep: `sdpa_bench2.sh`. Long cache: `sdpa_bench4.sh OUT 7 timed` with `sdpa_bench_far.py`. Speed: `swscreen5.sh OUT 5 MODELDIR final PTOK` on the final tree and `swscreen4.sh OUT 5 MODELDIR sdpa` on commit 01e82560.

Numerics: `sdpa-num.sh OUT MODELDIR short|long`, `sdpa-numref.sh OUT MODELDIR`, `sdpa-reflong.sh OUT MODELDIR BASEDIR SPLITDIR` and `analyze-long.py`. Controls: `sdpa-attn32.sh`, `sdpa-attn32v.sh` (variants `rev` and `blk`), `sdpa-attn32p.sh` (variant `perm`) and `sdpa-ref32v.sh`. Per-op: `sdpa-bias.sh OUT TRIALS`. Divergence: `gen-div.sh OUT MODELDIR 2000`. Doctest: `final-doctest.sh build` and `run OUT`, and `sdpa-run-bin.sh OUT`. The 7600-token text is `tfref-text.txt` three times. Scripts in `scripts/` have lab paths and names neutralised.

## Reopen

Reopen this change if an unmeasured model shows a perplexity cost larger than the one above. Reopen it if the part thresholds move on a new Mesa build, or if a cache above 32768 keys loses speed. `MLX_OMARCHY_SDPA_DECODE_SPLIT=0` restores the old route at run time.
