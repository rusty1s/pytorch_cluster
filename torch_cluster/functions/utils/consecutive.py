import torch
from torch_unique import unique


def _get_type(max_value, cuda):
    if max_value <= 255:
        return torch.cuda.ByteTensor if cuda else torch.ByteTensor
    elif max_value <= 32767:  # pragma: no cover
        return torch.cuda.ShortTensor if cuda else torch.ShortTensor
    elif max_value <= 2147483647:  # pragma: no cover
        return torch.cuda.IntTensor if cuda else torch.IntTensor
    else:  # pragma: no cover
        return torch.cuda.LongTensor if cuda else torch.LongTensor


def consecutive(x):
    size = x.size()
    u = unique(x.view(-1))
    len = u[-1] + 1
    max = u.size(0)
    type = _get_type(max, x.is_cuda)
    arg = type(len)
    arg[u] = torch.arange(0, max, out=type(max))
    x = arg[x.view(-1)]
    x = x.view(size).long()
    return x
