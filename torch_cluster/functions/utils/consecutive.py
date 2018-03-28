import torch
from torch_unique import unique


def _get_type(max, cuda):
    if max <= 255:
        return torch.cuda.ByteTensor if cuda else torch.ByteTensor
    elif max <= 32767:  # pragma: no cover
        return torch.cuda.ShortTensor if cuda else torch.ShortTensor
    elif max <= 2147483647:  # pragma: no cover
        return torch.cuda.IntTensor if cuda else torch.IntTensor
    else:  # pragma: no cover
        return torch.cuda.LongTensor if cuda else torch.LongTensor


def consecutive(tensor, return_unique=False):
    size = tensor.size()
    u = unique(tensor.view(-1))
    len = u[-1] + 1
    max = u.size(0)
    type = _get_type(max, tensor.is_cuda)
    arg = type(len)
    arg[u] = torch.arange(0, max, out=type(max))
    tensor = arg[tensor.view(-1)]
    tensor = tensor.view(size).long()

    return (tensor, u) if return_unique else tensor
