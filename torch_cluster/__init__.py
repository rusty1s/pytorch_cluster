import importlib
import os.path as osp

import torch

__version__ = '1.5.7'

for library in [
        '_version', '_grid', '_graclus', '_fps', '_rw', '_sampler', '_nearest',
        '_knn', '_radius'
]:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        library, [osp.dirname(__file__)]).origin)

if torch.version.cuda is not None:  # pragma: no cover
    cuda_version = torch.ops.torch_cluster.cuda_version()

    if cuda_version == -1:
        major = minor = 0
    elif cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major or t_minor != minor:
        raise RuntimeError(
            f'Detected that PyTorch and torch_cluster were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_cluster has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_cluster that '
            f'matches your PyTorch install.')

from .graclus import graclus_cluster  # noqa
from .grid import grid_cluster  # noqa
from .fps import fps  # noqa
from .nearest import nearest  # noqa
from .knn import knn, knn_graph  # noqa
from .radius import radius, radius_graph  # noqa
from .rw import random_walk  # noqa
from .sampler import neighbor_sampler  # noqa

__all__ = [
    'graclus_cluster',
    'grid_cluster',
    'fps',
    'nearest',
    'knn',
    'knn_graph',
    'radius',
    'radius_graph',
    'random_walk',
    'neighbor_sampler',
    '__version__',
]
