# TensorFold on omarchy-mlx

Status: in progress as of 2026-10-08.

[TensorFold](https://github.com/ashhart/TensorFold) is an inference
engine for quantized language models. We run its Python engine line,
pinned to v0.6.5, on the omarchy-mlx Vulkan backend. Upstream moved
that line to the `python-0.6` branch, so the pin stays where it is.

## What works

The port builds and runs on Omarchy Linux. The int8 matmul path had a
defect that skipped most output rows on wide shapes; the fix is merged
on omarchy-mlx main. The kernel-level audit behind the port is in
[receipts/2026-10-04-tensorfold-audit/README.md](../receipts/2026-10-04-tensorfold-audit/README.md).

## What is left

The correctness run against reference output is still going. Until it
closes, TensorFold stays listed as in progress in the
[README](../README.md#what-runs-on-omarchy-linux), and this page does
not give install steps.
