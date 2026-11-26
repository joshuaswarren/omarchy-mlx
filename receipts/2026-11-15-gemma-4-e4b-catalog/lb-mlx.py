import os, time, json
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
D = os.path.expanduser('~/lb-model/mlx-gemma4-e4b')
print('device', mx.default_device())
t0 = time.time()
model, tok = load(D)
print('load_s', round(time.time() - t0, 1))
msgs = [{'role': 'system', 'content': 'You answer briefly from survival knowledge.'},
        {'role': 'user', 'content': 'Name three ways to make water safe to drink in an emergency.'}]
try:
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
except TypeError:
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
print('prompt', repr(p[-120:]))
for i in range(3):
    last, text = None, ''
    for r in stream_generate(model, tok, p, max_tokens=120, sampler=make_sampler(temp=0.0)):
        last = r; text += r.text
    print('run', i, 'prompt_tokens', last.prompt_tokens, 'prompt_tps', round(last.prompt_tps, 1), 'gen_tokens', last.generation_tokens,
          'gen_tps', round(last.generation_tps, 2), 'peak_gb', round(last.peak_memory, 2))
    print('text', text[:160].replace('\n', ' '))
