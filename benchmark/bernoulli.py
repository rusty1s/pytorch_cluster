import time

import torch
from torch_cluster.functions.utils.ffi import _get_func

output = torch.cuda.FloatTensor(500000000).fill_(0.5)
torch.cuda.synchronize()
t = time.perf_counter()
torch.bernoulli(output)
torch.cuda.synchronize()
print(time.perf_counter() - t)

output = output.long().fill_(-1)
func = _get_func('serial', output)
torch.cuda.synchronize()
t = time.perf_counter()
func(output, output, output, output)
torch.cuda.synchronize()
print(time.perf_counter() - t)
