"""One elementwise add on the named stream (gpu or cpu): the counter's control."""
import sys

import mlx.core as mx

device = {"gpu": mx.gpu, "cpu": mx.cpu}[sys.argv[1]]
a = mx.ones((4096,))
b = mx.add(a, a, stream=device)
mx.eval(b)
print(sys.argv[1], "sum", b.sum().item())
