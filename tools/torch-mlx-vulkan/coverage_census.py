#!/usr/bin/env python3
"""Lowering-coverage census: the aten op families mlx_vulkan does not
lower yet. Diff torch.ops.aten base names against _FUNC_SPECS keys,
drop the internal/back-end/structural noise, and print the survivors
grouped into the families lanes pick pieces from.

Run from this directory (or set PYTHONPATH to it); needs torch + mlx.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import mlx_vulkan

# Non-aten aliases in _FUNC_SPECS that should not count toward coverage.
_ALIAS = {"concat", "truediv", "true_divide", "getitem", "einsum_fn",
          "tensor", "to", "interpolate", "count_nonzero",
          "scalar_tensor", "as_tensor", "numel", "normalize",
          "rms_norm", "meshgrid", "dropout", "index", "where", "Float",
          "lift_fresh_copy", "alias", "detach", "clone", "contiguous",
          "size", "item", "unbind", "narrow", "reshape", "flatten",
          "squeeze", "squeeze_", "unsqueeze", "expand", "stack",
          "split", "split_with_sizes", "chunk", "select", "slice",
          "mean", "sum", "max", "min", "zeros", "zeros_like", "ones",
          "ones_like", "full", "full_like", "eye", "linspace", "arange",
          "tril", "pad", "tile", "repeat", "movedim", "moveaxis",
          "swapaxes", "permute", "t", "transpose", "matmul", "bincount"}

# Internal/back-end/structural aten names that are out of lowering scope
# (autograd plumbing, backend-specific paths, sparse/nested/quantized,
# sym-* introspection, fused/foreach kernels, RNN cells, weight packing).
_OUT_OF_SCOPE = re.compile(
    r"(^_|cudnn|miopen|mkldnn|quantiz|sparse|nested|jagged|sym_|philox"
    r"|foreach|fused|thnn|weight_int|_pack|_dyn|resize|storage"
    r"|is_pinned|pin_memory|is_coalesced|is_contiguous|is_same_size"
    r"|is_strides|sparse_dim|dense_dim|is_complex|to_mkldnn|lift$|^dim$"
    r"|^name$|^stride$|^item$|^zero$|^fill$|^copy$|^set_$|_backward$)")

# Families as lane pieces: (label, aten base names). Order is the working
# queue order — claimed families carry their open-PR number in the doc.
FAMILIES = [
    ("sort (claimed: mlx#1188 wheel half)", ["sort", "argsort", "msort"]),
    ("fmod/trunc/lerp (claimed: mlx#1196)", ["fmod", "fmod_", "trunc",
                                           "trunc_", "lerp", "lerp_"]),
    ("index_add/copy/fill (claimed: mlx#1199)",
     ["index_add", "index_add_", "index_copy", "index_copy_",
      "index_fill", "index_fill_"]),
    ("as_strided (claimed: mlx#1193)", ["as_strided", "as_strided_",
                                        "as_strided_copy",
                                        "as_strided_scatter"]),
    ("fftn family (claimed: mlx#1200)", ["fft_fftn", "fft_ifftn",
                                         "fft_fftshift", "fft_ifftshift"]),
    ("amin/amax/prod (claimed: mlx#1194)", ["amin", "amax", "aminmax",
                                            "prod", "nanmedian", "median",
                                            "nansum"]),
    ("cumprod/cummax/cummin (claimed: mlx#1184)",
     ["cumprod", "cumprod_", "cummax", "cummin", "logcumsumexp"]),
    ("isin (claimed: mlx#1179)", ["isin"]),
    ("activation forwards", ["leaky_relu", "leaky_relu_", "hardtanh",
                             "hardtanh_", "relu6", "prelu", "threshold",
                             "threshold_"]),
    ("activation backwards", ["leaky_relu_backward",
                              "hardtanh_backward", "glu_backward",
                              "hardsigmoid_backward",
                              "hardswish_backward", "mish_backward",
                              "sigmoid_backward", "softplus_backward",
                              "log_sigmoid_forward",
                              "log_sigmoid_backward", "logit_backward",
                              "elu_backward", "batch_norm_backward",
                              "native_group_norm_backward",
                              "unfold_backward", "diagonal_backward"]),
    ("scatter/index_put", ["scatter", "scatter_", "scatter_add",
                           "scatter_add_", "scatter_reduce",
                           "scatter_reduce_", "index_put", "index_put_",
                           "index_reduce", "index_reduce_",
                           "masked_scatter", "masked_scatter_",
                           "select_scatter", "slice_scatter",
                           "diagonal_scatter"]),
    ("log/exp extras", ["log2", "log10", "logit", "logaddexp2",
                        "logspace", "exp2", "square", "log_normal"]),
    ("trig extras", ["tan", "cosh_", "deg2rad", "rad2deg", "sinc",
                     "arccos", "arcsin", "arctan", "arctan2",
                     "arccosh", "arcsinh", "arctanh"]),
    ("sign/copysign family", ["sgn", "signbit", "heaviside", "copysign",
                              "frexp", "ldexp", "nextafter", "float_power",
                              "positive", "fix", "absolute"]),
    ("comparison extras", ["greater_equal", "less_equal", "logical_xor",
                           "logical_xor_", "isneginf", "isposinf",
                           "bitwise_not", "bitwise_left_shift",
                           "bitwise_right_shift"]),
    ("unique/set", ["unique_consecutive", "unique_dim", "one_hot",
                    "bucketize", "searchsorted"]),
    ("random factories", ["rand", "randn", "randint", "randperm",
                          "rand_like", "randn_like", "randint_like",
                          "normal", "uniform", "bernoulli", "poisson",
                          "cauchy", "exponential", "geometric",
                          "log_normal", "multinomial", "rrelu_with_noise"]),
    ("pad family", ["reflection_pad1d", "reflection_pad2d",
                    "reflection_pad3d", "replication_pad1d",
                    "replication_pad2d", "replication_pad3d",
                    "pad_sequence", "constant_pad_nd"]),
    ("view/copy extras", ["diagonal", "diagonal_copy", "narrow_copy",
                          "view_copy", "permute_copy", "transpose_copy",
                          "t_copy", "squeeze_copy", "unsqueeze_copy",
                          "expand_copy", "unbind_copy", "tensor_split",
                          "unsafe_split", "unsafe_split_with_sizes",
                          "unsafe_chunk", "split_with_sizes_copy",
                          "alias_copy", "empty_strided", "empty_permuted",
                          "new_empty_strided", "copy_",
                          "broadcast_tensors", "masked_select",
                          "nonzero", "nonzero_numpy", "nonzero_static"]),
    ("upsample", ["upsample_nearest1d", "upsample_nearest2d",
                  "upsample_nearest2d_backward", "upsample_nearest3d",
                  "upsample_bilinear2d", "upsample_bicubic2d",
                  "upsample_linear1d", "upsample_trilinear3d"]),
    ("pool extras", ["avg_pool3d", "avg_pool3d_backward",
                     "max_pool3d_with_indices",
                     "max_pool3d_with_indices_backward",
                     "max_unpool2d", "max_unpool3d",
                     "fractional_max_pool2d", "adaptive_max_pool2d",
                     "adaptive_max_pool2d_backward",
                     "adaptive_max_pool3d",
                     "adaptive_max_pool3d_backward"]),
    ("fft extras", ["fft_hfft", "fft_ihfft", "fft_hfft2", "fft_ihfft2",
                    "fft_hfftn", "fft_ihfftn", "fft_rfft2", "fft_irfft2",
                    "fft_rfftn", "fft_irfftn", "stft", "istft",
                    "hann_window"]),
    ("losses", ["binary_cross_entropy",
                "binary_cross_entropy_backward",
                "binary_cross_entropy_with_logits", "huber_loss",
                "huber_loss_backward", "smooth_l1_loss",
                "smooth_l1_loss_backward", "soft_margin_loss",
                "soft_margin_loss_backward", "hinge_embedding_loss",
                "margin_ranking_loss", "multi_margin_loss",
                "multilabel_margin_loss_forward", "pairwise_distance",
                "dist", "nll_loss2d_forward", "nll_loss2d_backward",
                "mse_loss_backward", "native_dropout",
                "native_dropout_backward", "alpha_dropout"]),
    ("activation extras", ["hardtanh", "prelu", "rrelu_with_noise",
                           "relu6", "native_dropout", "threshold"]),
    ("linalg", ["linalg_qr", "linalg_eig", "linalg_eigvals",
                "linalg_lu", "linalg_lu_factor_ex", "linalg_lu_solve",
                "linalg_inv_ex", "linalg_solve_triangular",
                "linalg_matrix_exp", "linalg_cross",
                "linalg_householder_product", "linalg_ldl_factor_ex",
                "linalg_ldl_solve", "linalg_polar", "linalg_matrix_sqrth",
                "svd", "lu_unpack", "ormqr", "triangular_solve",
                "cholesky_solve", "cholesky_inverse", "trace", "renorm",
                "linear_backward"]),
    ("special functions", ["erfc", "erfinv", "digamma", "lgamma",
                           "mvlgamma", "polygamma", "igamma", "igammac",
                           "i0", "i1"] +
     [n for n in dir(torch.ops.aten) if n.startswith("special_")]),
    ("complex", ["complex", "conj", "conj_physical", "real", "imag",
                 "angle", "polar", "view_as_complex", "view_as_real"]),
    ("misc math", ["nan_to_num", "hypot", "gcd", "lcm", "xlogy_",
                   "divide", "multiply", "subtract", "true_divide",
                   "equal", "ne_", "eq_", "ge_", "gt_", "le_", "lt_",
                   "mul_", "div_", "add_", "sub_", "pow_", "sqrt_",
                   "neg_", "abs_", "floor_", "ceil_", "round_",
                   "clamp_min_", "clamp_max_", "clip_", "remainder_",
                   "sigmoid_", "tanh_", "relu_", "exp_", "exp2_", "log_",
                   "log10_", "log2_", "log1p_", "expm1_", "sin_", "cos_",
                   "tan_", "sinh_", "cosh_", "asin_", "acos_", "atan_",
                   "atan2_", "asinh_", "acosh_", "atanh_", "erf_", "i0_",
                   "sign_", "sgn_", "reciprocal_", "rsqrt_", "square_"]),
    ("shape ops", ["block_diag", "diag", "diag_embed", "tril_indices",
                   "triu_indices", "pixel_shuffle", "pixel_unshuffle",
                   "channel_shuffle", "im2col", "col2im", "unfold",
                   "unfold_copy", "unfold_backward", "rot90",
                   "repeat_interleave"]),
    ("norm/loss windows", ["hann_window", "rrelu_with_noise_",
                           "rrelu_with_noise_functional"]),
    ("misc", ["nan_to_num_", "heaviside_", "copysign_", "frexp",
              "grid_sampler_2d", "grid_sampler_2d_backward",
              "grid_sampler_3d", "grid_sampler_3d_backward",
              "affine_grid_generator", "one_hot", "segment_reduce",
              "argwhere", "nonzero", "quantize_per_tensor", "to_dense",
              "masked_select", "bernoulli_", "uniform_", "normal_",
              "renorm_", "clamp_min_", "clamp_max_"]),
    ("rnn (large, likely defer)", ["lstm", "gru", "rnn_tanh", "rnn_relu",
                                   "quantized_gru", "quantized_lstm"]),
]


def main():
    lowered = set(mlx_vulkan._FUNC_SPECS.keys())
    aten = {n for n in dir(torch.ops.aten) if not n.startswith("__")}
    missing = sorted(aten - lowered - _ALIAS)
    in_scope = [m for m in missing if not _OUT_OF_SCOPE.search(m)]
    claimed = set()
    print(f"lowered keys: {len(lowered)}; aten base names missing: "
          f"{len(missing)}; in-scope after noise filter: {len(in_scope)}")
    for label, ops in FAMILIES:
        have = [o for o in ops if o in in_scope]
        claimed.update(have)
        if have:
            print(f"{label}: {' '.join(have)}")
    rest = [m for m in in_scope if m not in claimed]
    if rest:
        print("ungrouped leftovers:")
        print("  " + " ".join(rest))


if __name__ == "__main__":
    main()
