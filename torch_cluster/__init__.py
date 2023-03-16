import importlib
import os.path as osp

import torch

__version__ = '1.6.1'

for library in [
        '_version', '_grid', '_graclus', '_fps', '_rw', '_sampler', '_nearest',
        '_knn', '_radius'
]:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cuda', [osp.dirname(__file__)])
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")

cuda_version = torch.ops.torch_cluster.cuda_version()
if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torch_cluster were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_cluster has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_cluster that '
            f'matches your PyTorch install.')

from .fps import fps  # noqa
from .graclus import graclus_cluster  # noqa
from .grid import grid_cluster  # noqa
from .knn import knn, knn_graph  # noqa
from .nearest import nearest  # noqa
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
