# PT-P — maxpool2d/avgpool2d fwd+bwd lowering in mlx_vulkan

Row from Main (2026-11-23 7:31 PM CST): maxpool2d/avgpool2d fwd+bwd
lowering in the mlx_vulkan backend. Done when: zero pool fallbacks on
ResNet-50 fwd and CIFAR-10 train; ops named in receipt.

## Change (PR #797, head 2735704a1)

`tools/torch-mlx-vulkan/mlx_vulkan.py` — pure-mx lowerings, no new shader:

- fwd: strided window-stack — `K = kh*kw` strided slices (slicing with
  step over the padded input), `mx.stack(axis=-1)`, reduce (`max`/`sum`).
- bwd: zero-insert (each output element followed by `stride-1` zeros via
  pad+reshape) then per-offset shift/accumulate — handles stride<kernel
  overlap with no scatter/gather kernels.
- `mx_vulkan_train` backend added: `aot_autograd(fw,bw)` routes both
  subgraphs through the executor for train graphs. `mlx_vulkan` (inference)
  unchanged.
- `_ALWAYS_LOWER`: pool ops attempt lowering even when upstream produced
  torch tensors (pool bwd behind unlowered conv/bn bwd still stays on mlx).

## Ops named

fwd: `max_pool2d`, `max_pool2d_with_indices`, `avg_pool2d`,
`adaptive_avg_pool2d` (+aten `_adaptive_avg_pool2d`)
bwd: `max_pool2d_with_indices_backward`, `avg_pool2d_backward`,
`_adaptive_avg_pool2d_backward`
Args supported: stride (incl. aten empty-list default = kernel), padding,
dilation, ceil_mode, count_include_pad, divisor_override.

## Gates (jwm1 M1, wAY-ptu venv, MV_DT=fp32)

| gate | result |
|---|---|
| ResNet-50 fwd (torchvision weights=None, 1x3x224x224) | maxdiff 1.9e-05 vs CPU; pool fallbacks NONE |
| CIFAR-10 train step (conv-bn-relu-maxpool x2 + conv-bn-relu-avgpool + linear, 16x3x32x32, CE loss) | loss diff 2.4e-07; grad maxdiff 2.8e-08; pool fallbacks NONE |
| Op parity k2/k3, s1-s3, p0-1, ceil_mode t/f | fwd 0.0 exact; bwd <=3e-7 |

Pool ops LOWERED in logs: `max_pool2d_with_indices`,
`max_pool2d_with_indices_backward`, `avg_pool2d`, `avg_pool2d_backward`,
`adaptive_avg_pool2d`. Non-pool fallbacks remaining (out of scope):
`convolution_backward`, `native_batch_norm_backward`,
`threshold_backward`, `batch_norm`, `relu`, `iadd`, `flatten`, `t`,
`detach`, `boolean_dispatch`.

Test scripts: jwm1 `~/scratch/w70-ptp/{ptp_test.py,ptp_test2.py,ptp_gate.py}`;
backend staged at `~/scratch/w70-ptp/mlx_vulkan.py`.

## w74 CHANGES (comment 7121749267) — fixed at 66fe6665e

- `ceil_mode` overrun crash: `_pool_windows` growth-pads bottom/right
  to cover the farthest ceil window; `pad_value` (-inf max / 0 avg).
- Divisor fix under ceil overrun: avg divisor = window ∩ [0,Hp) for
  both include_pad values (ones pad 1/0), matching ATen clamp.
- `adaptive_avg_pool2d` None entries resolve to input dim; identity
  shortcut; bwd shortcuts identically.
- Test: `tools/torch-mlx-vulkan/test_pools.py` committed — w74 case
  (2,3,8,10 k3 s2 p1 ceil), w7W case (9x9 k2 s2 p0 ceil), k/s/p/ceil/
  include_pad sweep, adaptive None — 0 failures vs torch on jwm1.
- Gate re-run at 66fe6665e: ResNet-50 fwd 1.9e-05, CIFAR-10 train
  grad 2.8e-08, zero pool fallbacks. Re-review asked of w74.
