from typing import Any
import platform
import shutil

import torch

dtypes = [
    torch.half, torch.bfloat16, torch.float, torch.double, torch.int,
    torch.long
]
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    grad_dtypes = [torch.float, torch.double]
else:
    grad_dtypes = [torch.half, torch.float, torch.double]
floating_dtypes = grad_dtypes + [torch.bfloat16]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda:0')]


def tensor(x: Any, dtype: torch.dtype, device: torch.device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)


def triton_wrap(dt_device_seq):
    return [
        (dt, device, use_triton)
        for dt, device in dt_device_seq
        for use_triton in ([False, True] if device.type == 'cuda' else [False])
    ]


def has_compiler() -> bool:
    if platform.system().lower() != 'windows':
        return True
    return shutil.which('cl') is not None
