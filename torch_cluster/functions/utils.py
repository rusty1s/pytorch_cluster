import torch
from torch_unique import unique

from .._ext import ffi


def get_func(name, tensor):
    typename = type(tensor).__name__.replace('Tensor', '')
    cuda = 'cuda_' if tensor.is_cuda else ''
    func = getattr(ffi, 'cluster_{}_{}{}'.format(name, cuda, typename))
    return func


def consecutive(tensor):
    size = tensor.size()
    u = unique(tensor.view(-1))
    arg = torch.ByteTensor(u[-1])
    arg[u] = torch.arange(0, u.size(0), out=torch.ByteTensor())
    tensor = arg[tensor.view(-1)]
    return tensor.view(size).long()
