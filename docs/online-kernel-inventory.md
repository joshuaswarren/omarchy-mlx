# Online kernel inventory (post-eface087f6 lineage, SUP-379)

Reference table for every ABBA receipt: the online kernels shipped in the wheel, with the per-part default caps from the part-rule code. Each row cites the kernel's source path and the part-rule site.

| Kernel | Source path | Parts with default | Default cap / wiring |
|---|---|---|---|
| sdpa_decode_online | overlay/mlx/backend/omarchy/shaders/sdpa_decode_online.comp | G13G=128, G13C=320, G14C=320, G17C=320, G13S=192, G14G=192, G16S=320, Unmeasured=0 | k_len split default (part-rule; see compute.h part selection) |
| fast_rope | overlay/mlx/backend/omarchy/shaders/fast_rope*.comp | G16G | half-split rope default |
| fast_rope_norm | overlay/mlx/backend/omarchy/shaders/fast_rope_norm.comp | per part-rule (D<=128 fused variants, #958) | fused rope+norm default per part |
| fast_norm | overlay/mlx/backend/omarchy/shaders/fast_norm*.comp | default path | RMS/Norm fast builds |
| fast_norm_gated | overlay/mlx/backend/omarchy/shaders/fast_norm_gated*.comp | gated rows | gated RMS/Norm default |
| fast_norm_apple | overlay/mlx/backend/omarchy/shaders/fast_norm_apple*.comp | apple parts | apple norm default |
| fast_norm_vjp_dw | overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dw*.comp | vjp dw | vjp weight-grad default |
| fast_norm_vjp_dx | overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dx*.comp | vjp dx | vjp input-grad default |
| fast_norm_vjp_dw_reduce | overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dw_reduce*.comp | vjp dw reduce | vjp dw-reduce default |

Overrides and exact part-rule values: consult the part-selection code (compute.cpp / primitives.cpp part rules) — this table records the kernel set and the SDPA decode split caps measured in the #997 lineage (G13G=128, G13C=320, G14C=320, G17C=320, G13S=192, G14G=192, G16S=320, Unmeasured=0; RoPE half-split G16G only; GDN conv+norm fused G13G/G14C/G17C; vocab-tail chunked G14C/G17C; encoder earliest-wave G16S; SDPA causal flash coopmat G16S).
