# Per-pair ABBA data (generated from summary.json files on jw16)
# legs: a_ms/b_ms = median ms per generated token; tps = tokens/s; W = B faster

### cells/bk64_ctx1024
env_a: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=13.63ms b=14.27ms a_tok=73.37 b_tok=70.08 gain=-4.52% wins=0/3 ids=445cb2aab50cf666
  p1 AB: a=13.65ms/73.26tps b=14.26ms/70.12tps -4.29% L idmatch=True
  p2 BA: a=13.63ms/73.37tps b=14.28ms/70.05tps -4.52% L idmatch=True
  p3 AB: a=13.61ms/73.50tps b=14.27ms/70.08tps -4.65% L idmatch=True
### cells/bk64_ctx2048
env_a: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=14.62ms b=14.65ms a_tok=68.42 b_tok=68.24 gain=-0.13% wins=1/3 ids=9115b5280c1d7e33
  p1 AB: a=14.62ms/68.42tps b=14.67ms/68.16tps -0.38% L idmatch=True
  p2 BA: a=14.66ms/68.20tps b=14.65ms/68.24tps +0.07% W idmatch=True
  p3 AB: a=14.61ms/68.44tps b=14.63ms/68.35tps -0.13% L idmatch=True
### cells/bk64_ctx256
env_a: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=14.04ms b=14.04ms a_tok=71.25 b_tok=71.23 gain=-0.06% wins=0/3 ids=902e12e22d1d8367
  p1 AB: a=14.02ms/71.33tps b=14.03ms/71.29tps -0.06% L idmatch=True
  p2 BA: a=14.04ms/71.25tps b=14.04ms/71.23tps -0.02% L idmatch=True
  p3 AB: a=14.04ms/71.21tps b=14.05ms/71.16tps -0.07% L idmatch=True
### cells/bk64_ctx512
env_a: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=13.48ms b=13.03ms a_tok=74.16 b_tok=76.75 gain=+3.49% wins=3/3 ids=a6fad159dfa90047
  p1 AB: a=13.48ms/74.16tps b=13.03ms/76.75tps +3.49% W idmatch=True
  p2 BA: a=13.43ms/74.47tps b=13.03ms/76.77tps +3.09% W idmatch=True
  p3 AB: a=13.54ms/73.88tps b=13.05ms/76.63tps +3.72% W idmatch=True
### cells/bk64pv_ctx1024
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=13.65ms b=14.27ms a_tok=73.26 b_tok=70.05 gain=-4.42% wins=0/3 ids=445cb2aab50cf666
  p1 AB: a=13.66ms/73.22tps b=14.29ms/69.98tps -4.42% L idmatch=True
  p2 BA: a=13.65ms/73.26tps b=14.27ms/70.05tps -4.38% L idmatch=True
  p3 AB: a=13.60ms/73.52tps b=14.27ms/70.07tps -4.69% L idmatch=True
### cells/bk64pv_ctx2048
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=14.63ms b=15.91ms a_tok=68.37 b_tok=62.86 gain=-8.08% wins=0/3 ids=9115b5280c1d7e33
  p1 AB: a=14.63ms/68.34tps b=16.43ms/60.88tps -10.93% L idmatch=True
  p2 BA: a=14.62ms/68.40tps b=15.90ms/62.87tps -8.08% L idmatch=True
  p3 AB: a=14.63ms/68.37tps b=15.91ms/62.86tps -8.06% L idmatch=True
### cells/bk64pv_ctx256
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=14.05ms b=14.02ms a_tok=71.19 b_tok=71.31 gain=+0.18% wins=2/3 ids=902e12e22d1d8367
  p1 AB: a=14.04ms/71.23tps b=14.08ms/71.00tps -0.32% L idmatch=True
  p2 BA: a=14.06ms/71.11tps b=14.02ms/71.31tps +0.28% W idmatch=True
  p3 AB: a=14.05ms/71.19tps b=14.02ms/71.32tps +0.18% W idmatch=True
### cells/bk64pv_ctx512
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=13.44ms b=13.02ms a_tok=74.43 b_tok=76.79 gain=+3.17% wins=3/3 ids=a6fad159dfa90047
  p1 AB: a=13.43ms/74.48tps b=13.05ms/76.65tps +2.91% W idmatch=True
  p2 BA: a=13.47ms/74.24tps b=13.02ms/76.81tps +3.46% W idmatch=True
  p3 AB: a=13.44ms/74.43tps b=13.02ms/76.79tps +3.17% W idmatch=True
### cells/pv_ctx1024
env_a: {"MLX_OMARCHY_SDPA_SPLIT_PV": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=14.39ms b=13.63ms a_tok=69.49 b_tok=73.38 gain=+4.96% wins=3/3 ids=445cb2aab50cf666
  p1 AB: a=14.30ms/69.92tps b=13.63ms/73.39tps +4.96% W idmatch=True
  p2 BA: a=14.39ms/69.49tps b=13.63ms/73.38tps +5.60% W idmatch=True
  p3 AB: a=14.41ms/69.41tps b=13.78ms/72.56tps +4.55% W idmatch=True
### cells/pv_ctx2048
env_a: {"MLX_OMARCHY_SDPA_SPLIT_PV": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=16.02ms b=17.50ms a_tok=62.41 b_tok=57.13 gain=-8.71% wins=0/3 ids=9115b5280c1d7e33
  p1 AB: a=16.02ms/62.41tps b=16.56ms/60.37tps -3.27% L idmatch=True
  p2 BA: a=15.98ms/62.58tps b=17.50ms/57.13tps -8.71% L idmatch=True
  p3 AB: a=16.29ms/61.38tps b=21.00ms/47.62tps -22.42% L idmatch=True
### cells/pv_ctx256
env_a: {"MLX_OMARCHY_SDPA_SPLIT_PV": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=15.36ms b=15.22ms a_tok=65.12 b_tok=65.69 gain=+8.42% wins=2/3 ids=902e12e22d1d8367
  p1 AB: a=15.13ms/66.09tps b=16.72ms/59.80tps -9.52% L idmatch=True
  p2 BA: a=16.65ms/60.06tps b=15.22ms/65.69tps +9.38% W idmatch=True
  p3 AB: a=15.36ms/65.12tps b=14.16ms/70.61tps +8.42% W idmatch=True
### cells/pv_ctx512
env_a: {"MLX_OMARCHY_SDPA_SPLIT_PV": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=14.26ms b=13.58ms a_tok=70.10 b_tok=73.62 gain=+5.02% wins=3/3 ids=a6fad159dfa90047
  p1 AB: a=14.33ms/69.80tps b=13.55ms/73.82tps +5.76% W idmatch=True
  p2 BA: a=14.26ms/70.10tps b=13.58ms/73.62tps +5.02% W idmatch=True
  p3 AB: a=14.24ms/70.21tps b=13.84ms/72.27tps +2.95% W idmatch=True
### cells/splitall_ctx1024
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=13.73ms b=14.33ms a_tok=72.81 b_tok=69.79 gain=-4.60% wins=0/3 ids=445cb2aab50cf666
  p1 AB: a=13.67ms/73.16tps b=14.33ms/69.79tps -4.60% L idmatch=True
  p2 BA: a=13.77ms/72.60tps b=14.32ms/69.82tps -3.83% L idmatch=True
  p3 AB: a=13.73ms/72.81tps b=14.46ms/69.15tps -5.03% L idmatch=True
### cells/splitall_ctx2048
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=14.74ms b=16.08ms a_tok=67.84 b_tok=62.19 gain=-7.80% wins=0/3 ids=9115b5280c1d7e33
  p1 AB: a=14.66ms/68.23tps b=16.14ms/61.98tps -9.17% L idmatch=True
  p2 BA: a=14.94ms/66.94tps b=16.08ms/62.19tps -7.10% L idmatch=True
  p3 AB: a=14.74ms/67.84tps b=15.99ms/62.55tps -7.80% L idmatch=True
### cells/splitall_ctx256
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=15.25ms b=12.90ms a_tok=65.57 b_tok=77.53 gain=+14.75% wins=3/3 ids=902e12e22d1d8367
  p1 AB: a=14.13ms/70.77tps b=12.89ms/77.56tps +9.59% W idmatch=True
  p2 BA: a=18.19ms/54.96tps b=12.90ms/77.53tps +41.06% W idmatch=True
  p3 AB: a=15.25ms/65.57tps b=13.29ms/75.25tps +14.75% W idmatch=True
### cells/splitall_ctx512
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=13.49ms b=13.07ms a_tok=74.15 b_tok=76.50 gain=+3.10% wins=3/3 ids=a6fad159dfa90047
  p1 AB: a=13.45ms/74.35tps b=13.07ms/76.50tps +2.88% W idmatch=True
  p2 BA: a=13.49ms/74.15tps b=13.08ms/76.45tps +3.10% W idmatch=True
  p3 AB: a=13.50ms/74.05tps b=13.06ms/76.60tps +3.44% W idmatch=True
### confirm/bk64_ctx512
env_a: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=13.54ms b=13.10ms a_tok=73.86 b_tok=76.31 gain=+3.25% wins=5/5 ids=a6fad159dfa90047
  p1 AB: a=13.44ms/74.41tps b=13.10ms/76.31tps +2.56% W idmatch=True
  p2 BA: a=13.52ms/73.97tps b=13.12ms/76.22tps +3.05% W idmatch=True
  p3 AB: a=13.57ms/73.69tps b=13.14ms/76.08tps +3.25% W idmatch=True
  p4 BA: a=13.65ms/73.23tps b=13.09ms/76.41tps +4.34% W idmatch=True
  p5 AB: a=13.54ms/73.86tps b=13.07ms/76.50tps +3.57% W idmatch=True
### confirm/bk64pv_ctx512
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_BK64": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=15.19ms b=17.00ms a_tok=65.82 b_tok=58.81 gain=+0.60% wins=3/5 ids=a6fad159dfa90047
  p1 AB: a=15.19ms/65.82tps b=13.91ms/71.89tps +9.21% W idmatch=True
  p2 BA: a=20.66ms/48.39tps b=21.19ms/47.18tps -2.50% L idmatch=True
  p3 AB: a=17.66ms/56.62tps b=17.56ms/56.96tps +0.60% W idmatch=True
  p4 BA: a=13.51ms/74.02tps b=17.00ms/58.81tps -20.55% L idmatch=True
  p5 AB: a=13.43ms/74.44tps b=13.04ms/76.70tps +3.04% W idmatch=True
### confirm/pv_ctx1024
env_a: {"MLX_OMARCHY_SDPA_SPLIT_PV": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=14.36ms b=13.69ms a_tok=69.63 b_tok=73.04 gain=+4.80% wins=5/5 ids=445cb2aab50cf666
  p1 AB: a=14.36ms/69.63tps b=13.65ms/73.26tps +5.22% W idmatch=True
  p2 BA: a=14.34ms/69.72tps b=13.63ms/73.37tps +5.24% W idmatch=True
  p3 AB: a=14.41ms/69.38tps b=13.80ms/72.44tps +4.41% W idmatch=True
  p4 BA: a=14.38ms/69.56tps b=13.75ms/72.74tps +4.57% W idmatch=True
  p5 AB: a=14.35ms/69.70tps b=13.69ms/73.04tps +4.80% W idmatch=True
### confirm/pv_ctx256
env_a: {"MLX_OMARCHY_SDPA_SPLIT_PV": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=14.09ms b=14.12ms a_tok=70.98 b_tok=70.83 gain=-0.30% wins=2/5 ids=902e12e22d1d8367
  p1 AB: a=14.12ms/70.81tps b=14.09ms/70.96tps +0.23% W idmatch=True
  p2 BA: a=14.09ms/70.98tps b=14.14ms/70.72tps -0.37% L idmatch=True
  p3 AB: a=14.14ms/70.73tps b=14.10ms/70.94tps +0.29% W idmatch=True
  p4 BA: a=14.08ms/71.05tps b=14.12ms/70.83tps -0.30% L idmatch=True
  p5 AB: a=14.07ms/71.07tps b=14.14ms/70.73tps -0.47% L idmatch=True
### confirm/pv_ctx512
env_a: {"MLX_OMARCHY_SDPA_SPLIT_PV": "0"}  env_b: {"MLX_OMARCHY_SDPA_SPLIT_PV": "1"}
median a=14.23ms b=13.50ms a_tok=70.29 b_tok=74.08 gain=+5.37% wins=5/5 ids=a6fad159dfa90047
  p1 AB: a=14.25ms/70.20tps b=13.49ms/74.12tps +5.58% W idmatch=True
  p2 BA: a=14.21ms/70.37tps b=13.49ms/74.12tps +5.32% W idmatch=True
  p3 AB: a=14.23ms/70.29tps b=13.58ms/73.66tps +4.79% W idmatch=True
  p4 BA: a=14.37ms/69.61tps b=13.54ms/73.88tps +6.14% W idmatch=True
  p5 AB: a=14.22ms/70.31tps b=13.50ms/74.08tps +5.37% W idmatch=True
### confirm/splitall_ctx256
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=14.09ms b=12.90ms a_tok=70.99 b_tok=77.53 gain=+9.32% wins=5/5 ids=902e12e22d1d8367
  p1 AB: a=14.08ms/71.04tps b=12.91ms/77.44tps +9.02% W idmatch=True
  p2 BA: a=14.11ms/70.89tps b=12.90ms/77.50tps +9.32% W idmatch=True
  p3 AB: a=14.07ms/71.07tps b=12.90ms/77.53tps +9.10% W idmatch=True
  p4 BA: a=14.10ms/70.92tps b=12.87ms/77.73tps +9.60% W idmatch=True
  p5 AB: a=14.09ms/70.99tps b=12.88ms/77.63tps +9.34% W idmatch=True
### confirm/splitall_ctx512
env_a: {}  env_b: {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "1"}
median a=13.47ms b=13.04ms a_tok=74.27 b_tok=76.68 gain=+3.29% wins=5/5 ids=a6fad159dfa90047
  p1 AB: a=13.46ms/74.32tps b=13.02ms/76.83tps +3.37% W idmatch=True
  p2 BA: a=13.47ms/74.27tps b=13.06ms/76.59tps +3.13% W idmatch=True
  p3 AB: a=13.49ms/74.15tps b=13.05ms/76.60tps +3.31% W idmatch=True
  p4 BA: a=13.47ms/74.24tps b=13.04ms/76.68tps +3.28% W idmatch=True
  p5 AB: a=13.45ms/74.36tps b=13.02ms/76.81tps +3.29% W idmatch=True
