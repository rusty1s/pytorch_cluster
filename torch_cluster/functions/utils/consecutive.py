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
    initial_size = x.size()

    # Compute unique vector.
    u = unique(x.view(-1))

    # Compute mask with mask[u[0]] = 0, mask[u[1]] = 1, ...
    # As mask can get very big (dependent on the maximum value in `x`, we want
    # to take the least possible amount of space on disk (`_get_type`).
    max_value = u[-1] + 1
    mask = _get_type(u.size(0), x.is_cuda)(max_value)
    mask[u] = torch.arange(0, u.size(0), out=mask.new())

    # Select the values in `mask` based on `x` and reshape to initial size.
    x = mask[x.view(-1)]
    x = x.view(initial_size)
    x = x.long()

    return x
