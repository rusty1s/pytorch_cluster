import torch
from torch_unique import unique

from .._ext import ffi


def get_func(name, tensor):
    typename = type(tensor).__name__.replace('Tensor', '')
    cuda = 'cuda_' if tensor.is_cuda else ''
    func = getattr(ffi, 'cluster_{}_{}{}'.format(name, cuda, typename))
    return func


def get_type(max, cuda):
    if max <= 255:
        return torch.cuda.ByteTensor if cuda else torch.ByteTensor
    elif max <= 32767:  # pragma: no cover
        return torch.cuda.ShortTensor if cuda else torch.ShortTensor
    elif max <= 2147483647:  # pragma: no cover
        return torch.cuda.IntTensor if cuda else torch.IntTensor
    else:  # pragma: no cover
        return torch.cuda.LongTensor if cuda else torch.LongTensor


def consecutive(tensor, return_batch=None):
    size = tensor.size()
    u = unique(tensor.view(-1))
    len = u[-1] + 1
    max = u.size(0)
    type = get_type(max, tensor.is_cuda)
    arg = type(len)
    arg[u] = torch.arange(0, max, out=type(max))
    tensor = arg[tensor.view(-1)]
    return tensor.view(size).long(), u
