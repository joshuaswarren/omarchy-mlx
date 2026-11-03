"""Generate unrolled vs looped ALU kernels with the same dynamic op count."""
import random, sys
random.seed(1)
HDR = """#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer S { uint s[]; };
layout(set = 0, binding = 1) writeonly buffer O { uint o[]; };
layout(push_constant) uniform P { uint n; } p;
void main() {
  uint gid = gl_GlobalInvocationID.x;
  uint a = s[gid & 1023u], b = a ^ 0x9e3779b9u, c = a + 7u, d = b * 3u;
"""
for ops in (250, 500, 1000, 2000, 4000):
    body = []
    for i in range(ops):
        k = random.getrandbits(32) | 1
        v = "abcd"[i % 4]; w = "abcd"[(i + 1) % 4]
        body.append(f"  {v} = {v} * {k}u + ({w} ^ {random.getrandbits(32)}u);")
    open(f"icu_{ops}.comp", "w").write(HDR + "\n".join(body) + "\n  o[gid] = a ^ b ^ c ^ d;\n}\n")
    loop = f"""  for (uint i = 0u; i < {ops // 4}u; ++i) {{
    a = a * 2654435761u + (b ^ i);
    b = b * 2246822519u + (c ^ i);
    c = c * 3266489917u + (d ^ i);
    d = d * 668265263u + (a ^ i);
  }}
"""
    open(f"icl_{ops}.comp", "w").write(HDR + loop + "  o[gid] = a ^ b ^ c ^ d;\n}\n")
print("ok")
