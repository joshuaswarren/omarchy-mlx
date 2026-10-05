# A8 DeepSeek draft: _get_dims cannot size DeepseekV2YarnRotaryEmbedding

Status: DRAFT — upstream model-class gap; no omarchy-side fix planned.

## Finding

omlx/patches/specprefill.py:949-958 (`_get_dims`) probes `_dims`, `dim`,
`dims` on the genuine rope module and raises
`ValueError: Cannot determine dims from <class
'mlx_lm.models.deepseek_v2.DeepseekV2YarnRotaryEmbedding'>` when none
exist. Observed on the M2 (artifacts/OmlxCache/omlx-cache-rows/
a8-20261005T135842Z/server.log:98,139), SpecPrefill + DeepSeek-Coder-V2-
Lite-Instruct-4bit.

The class (mlx-lm deepseek_v2.py, `class DeepseekV2YarnRotaryEmbedding`)
takes `dim` as a constructor argument but stores only `self.mscale` and
computed frequency tensors — it never keeps `dim`/`dims` as an instance
attribute. This is the stock mlx-lm class, so the failure is
backend-INDEPENDENT: stock macOS oMLX v0.7.0 would hit the same
ValueError for DeepSeek (only the Qwen3-4B pair was contrasted on macOS;
it passes there because the qwen3 rope carries `.dims`).

## Fix surface (upstream / omlx, not omarchy)

Either mlx-lm stores `self.dims = dim` in the yarn class, or omlx
`_get_dims` gains a type-keyed branch for DeepseekV2YarnRotaryEmbedding
(`dims = int(head_dim)` — in deepseek_v2 the rope width equals the head
dim, no partial rotary). Until then the A8 row stays documented as
fallback-on-DeepSeek with the Qwen3-4B evidence carrying the row.

Related: the Qwen3-4B half of A8 is FIXED omarchy-side (wrapped-rope
fallback gate, commit c48c0d458).
