# Draft: oMLX v0.7.0 benchmark wording discrepancy

Scope: documentation note only; no upstream issue or PR opened.

The pinned README advertises the performance benchmark as measuring PP/TG with "partial prefix cache hit testing" (`README.md:253-260`). The pinned implementation does not provide that benchmark behavior: `omlx/admin/benchmark.py` builds prompts with unique UUID prefixes (`:543-561`, `:590-601`), sets `skip_cache_store=True` for measured requests (`:759`, `:1103`), warns when a measured request unexpectedly sees cached tokens (`:816-820`), and creates unique prefixes for batch cases (`:2004-2008`). The benchmark panel request fields are `model_id`, `prompt_lengths`, `generation_length`, `batch_sizes`, `context_profile`, and warmup mode (`:109-118`). No partial-prefix-hit option was found in `_bench.html` or `benchmark.py`.

For omarchy-mlx parity, A17 should validate the shipped PP/TG panel and no-cache behavior. Partial prefix-cache behavior is a separate feature owned under A5. Do not count this README/code mismatch as an omarchy defect or file it upstream without separate approval.

Status: draft; factual check against pinned `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`; no hardware PP/TG run attached yet.
