"""Regenerate muse_tiny_reference.json: reference logits of the transformers MuseSparkTextModel (fp32, CPU) for the tiny weights.

Needs torch and transformers >= 5.20.0 (the release that has the muse_spark model). The test does not run this; it reads the JSON.

    python3 tests/fixtures/muse/generate_fixture.py

The head applies the same scaling the transformers MuseSparkForConditionalGeneration applies after lm_head: logits times output_multiplier,
divided by final_logit_softcapping, tanh, times final_logit_softcapping.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.muse_spark import configuration_muse_spark as cc
from transformers.models.muse_spark import modeling_muse_spark as mm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weights  # noqa: E402

cfg = cc.MuseSparkTextConfig(**weights.CONFIG)
cfg._attn_implementation = "eager"
model = mm.MuseSparkTextModel(cfg).eval()
state, lm_head, ids = weights.make_weights()

have = {k: tuple(v.shape) for k, v in model.state_dict().items()}
want = {k: tuple(v.shape) for k, v in state.items()}
if have != want:
    raise SystemExit(f"name or shape mismatch: missing {sorted(set(have) - set(want))}, extra {sorted(set(want) - set(have))}")
model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
with torch.no_grad():
    hidden = model(input_ids=torch.tensor(ids)[None], use_cache=False).last_hidden_state
    z = (hidden @ torch.from_numpy(lm_head).T) * cfg.output_multiplier / cfg.final_logit_softcapping
    logits = (torch.tanh(z) * cfg.final_logit_softcapping)[0].numpy()

out = {
    "config": weights.CONFIG,
    "seed": weights.SEED,
    "head_scale": weights.HEAD_SCALE,
    "weights_sha256": weights.weights_digest(state, lm_head),
    "ids": [int(i) for i in ids],
    "reference": {"implementation": "transformers MuseSparkTextModel", "transformers": transformers.__version__,
                  "torch": torch.__version__, "numpy": np.__version__, "dtype": "float32", "device": "cpu"},
    "layer_types": list(cfg.layer_types),
    "logits": [[float(x) for x in row] for row in logits],
}
path = HERE / "muse_tiny_reference.json"
path.write_text(json.dumps(out, indent=0) + "\n")
print("wrote", path, "positions", logits.shape[0], "vocab", logits.shape[1], "abs max", float(np.abs(logits).max()))
