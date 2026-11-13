#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Rewrite the Parakeet CoreML encoder MIL into a program ANECCompile accepts
for h16g, with the mask semantics of the original graph.

ANECCompile refuses the encoder as exported: its sequence-length arithmetic
runs on int32 and bool tensors. Every mask in it is a function of the input
mask, which is a prefix (valid frames first). After each stride-2
subsampling conv (kernel 3, padding 1) the length is ceil(L / 2), so frame i
stays valid exactly when 2i < L, i.e. when mask[2i] is 1, and after three of
them when mask[8i] is 1. The rewrite therefore takes the mask as an fp16
input (1 valid, 0 padding) and replaces the length arithmetic by strided
slices of it; the 375 x 375 attention mask is the fp16 outer product of the
375-frame mask compared with 0.5, and the rows without a valid key are the
padded query rows. encoder_hidden is the only output: the H16 driver takes
one output per program, and encoder_mask is computed on the host.

Measured on the Mac16,7 ANE against the original model on an Apple ANE
(CoreML, same h16 generation): encoder_hidden is bit-identical for a full
mask and for a 504-frame duration-aware mask, padded rows included.

  h16_encoder_mil.py ENCODER.mil OUT.mil
"""
import re
import sys

src = open(sys.argv[1]).read()
head, rest = src.split('func main<ios18>(', 1)
sig, body = rest.split(') {', 1)
body, tail = body.rsplit('} -> (', 1)
outs_tail = tail
lines = [l for l in body.split('\n') if l.strip()]
stmt_re = re.compile(r'^\s*(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s+=\s+([A-Za-z_0-9.]+)\((.*)\)\[name = string\("[^"]*"\)(.*)\];\s*$')
ident = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')
stmts = []
for l in lines:
    m = stmt_re.match(l)
    if not m: sys.exit('unparsed: ' + l[:200])
    typ, name, op, args, extra = m.groups()
    stmts.append(dict(line=l, typ=typ, name=name, op=op, args=args))
redefine = {}
I = '            '
def S(typ, name, op, args): return f'{I}{typ} {name} = {op}({args})[name = string("{name}")];'
new = []
new.append(S('tensor<int32, [2]>', 'mx_begin', 'const', '') .replace('const()', 'const()'))
# consts via explicit const syntax
def C(typ, name, val): return f'{I}{typ} {name} = const()[name = string("{name}"), val = {val}];'
pre = [
 C('tensor<int32, [2]>', 'mx_begin', 'tensor<int32, [2]>([0, 0])'),
 C('tensor<int32, [2]>', 'mx_end', 'tensor<int32, [2]>([1, 3000])'),
 C('tensor<int32, [2]>', 'mx_s2', 'tensor<int32, [2]>([1, 2])'),
 C('tensor<int32, [2]>', 'mx_s4', 'tensor<int32, [2]>([1, 4])'),
 C('tensor<int32, [2]>', 'mx_s8', 'tensor<int32, [2]>([1, 8])'),
 S('tensor<fp16, [1, 1500]>', 'mx_m1500', 'slice_by_index', 'begin = mx_begin, end = mx_end, stride = mx_s2, x = attention_mask'),
 S('tensor<fp16, [1, 750]>', 'mx_m750', 'slice_by_index', 'begin = mx_begin, end = mx_end, stride = mx_s4, x = attention_mask'),
 S('tensor<fp16, [1, 375]>', 'mx_m375', 'slice_by_index', 'begin = mx_begin, end = mx_end, stride = mx_s8, x = attention_mask'),
 C('tensor<int32, [4]>', 'mx_shape1500', 'tensor<int32, [4]>([1, 1, 1500, 1])'),
 C('tensor<int32, [4]>', 'mx_shape750', 'tensor<int32, [4]>([1, 1, 750, 1])'),
 C('tensor<int32, [4]>', 'mx_shape375', 'tensor<int32, [4]>([1, 1, 375, 1])'),
 S('tensor<fp16, [1, 1, 1500, 1]>', 'var_116_to_fp16', 'reshape', 'shape = mx_shape1500, x = mx_m1500'),
 S('tensor<fp16, [1, 1, 750, 1]>', 'var_143_to_fp16', 'reshape', 'shape = mx_shape750, x = mx_m750'),
 S('tensor<fp16, [1, 1, 375, 1]>', 'var_187_to_fp16', 'reshape', 'shape = mx_shape375, x = mx_m375'),
 C('tensor<int32, [3]>', 'mx_col', 'tensor<int32, [3]>([1, 375, 1])'),
 C('tensor<int32, [3]>', 'mx_row', 'tensor<int32, [3]>([1, 1, 375])'),
 C('tensor<int32, [4]>', 'mx_sq', 'tensor<int32, [4]>([1, 1, 375, 375])'),
 S('tensor<fp16, [1, 375, 1]>', 'mx_q', 'reshape', 'shape = mx_col, x = mx_m375'),
 S('tensor<fp16, [1, 1, 375]>', 'mx_k', 'reshape', 'shape = mx_row, x = mx_m375'),
 S('tensor<fp16, [1, 375, 375]>', 'mx_qk', 'mul', 'x = mx_q, y = mx_k'),
 S('tensor<fp16, [1, 1, 375, 375]>', 'mx_qk4', 'reshape', 'shape = mx_sq, x = mx_qk'),
 C('fp16', 'mx_half', 'fp16(0x1p-1)'),
 S('tensor<bool, [1, 1, 375, 375]>', 'var_373', 'less', 'x = mx_qk4, y = mx_half'),
 S('tensor<bool, [1, 1, 375]>', 'all_masked_rows_1', 'less', 'x = mx_k, y = mx_half'),
]
redefined = {'var_116_to_fp16', 'var_143_to_fp16', 'var_187_to_fp16', 'var_373', 'all_masked_rows_1'}
alias = {'input_features_to_fp16': 'input_features'}
kept = []
for s in stmts:
    if s['name'] in redefined or s['name'] in alias: continue
    kept.append(s)
def refs(args): return set(ident.findall(re.sub(r'"[^"]*"', '', args)))
# substitute aliases in args
for s in kept:
    for a, b in alias.items():
        if re.search(r'\b' + a + r'\b', s['args']):
            s['line'] = re.sub(r'(= [A-Za-z_0-9.]+\()(.*)(\)\[name)', lambda m: m.group(1) + re.sub(r'\b' + a + r'\b', b, m.group(2)) + m.group(3), s['line'], count=1)
            s['args'] = re.sub(r'\b' + a + r'\b', b, s['args'])
allst = [dict(line=l, name=stmt_re.match(l).group(2), args=stmt_re.match(l).group(4)) for l in pre] + kept
outputs = ['encoder_hidden']
# encoder_hidden leaves the ANE in fp16: the fp32 cast at the end becomes an
# identity of its fp16 source (the host widens exactly).
for s in allst:
    if s['name'] == 'encoder_hidden':
        m = re.search(r'x = ([A-Za-z_][A-Za-z0-9_]*)\)', s['line'])
        if not m or ' = cast(' not in s['line']:
            sys.exit('encoder_hidden is not the final cast')
        s['line'] = (f'{I}tensor<fp16, [1, 375, 640]> encoder_hidden = '
                     f'identity(x = {m.group(1)})[name = string("encoder_hidden")];')
        s['args'] = f'x = {m.group(1)}'
# dead code elimination
while True:
    used = set(outputs)
    for s in allst: used |= refs(s['args'])
    before = len(allst); allst = [s for s in allst if s['name'] in used]
    if len(allst) == before: break
# verify no int32/bool data-path tensors remain except the two masks fed to select
bad = [s['line'].strip()[:160] for s in allst if re.match(r'\s*tensor<(int32|bool)', s['line']) and ' = const()' not in s['line'] and s['name'] not in ('var_373', 'all_masked_rows_1')]
# order: pre-statements must come before users; keep pre first then original order
sig_new = 'tensor<fp16, [1, 3000]> attention_mask, tensor<fp16, [1, 3000, 128]> input_features'
out = head + 'func main<ios18>(' + sig_new + ') {\n' + '\n'.join(s['line'] for s in allst) + '\n        } -> (encoder_hidden);\n}\n'
open(sys.argv[2], 'w').write(out)
print('statements', len(stmts), '->', len(allst), 'remaining int/bool data tensors:', bad[:10])
