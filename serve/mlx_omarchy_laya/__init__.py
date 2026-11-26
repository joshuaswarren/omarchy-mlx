"""mlx_omarchy_laya: Laya typed-decision models on the MLX Omarchy backend.

Laya (convaiinnovations/laya, Apache-2.0) is a non-autoregressive typed
decision model: a ModernBERT-large encoder plus a decision head that scores
per-option [MASK] markers and an escalate/answer action head. This package
ports the pinned upstream inference path (rl_common.py / rl_agent_api.py at
revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982) to pure mlx.core ops.

It never exposes chat completions: output_tokens is always zero and the wire
schema is upstream's system_one envelope.
"""

__version__ = "0.1.0"
