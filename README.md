[pypi-image]: https://badge.fury.io/py/torch-cluster.svg
[pypi-url]: https://pypi.python.org/pypi/torch-cluster
[build-image]: https://travis-ci.org/rusty1s/pytorch_cluster.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_cluster
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_cluster/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_cluster?branch=master

# PyTorch Cluster

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

--------------------------------------------------------------------------------

This package consists of a small extension library of highly optimized graph cluster algorithms for the use in [PyTorch](http://pytorch.org/).
The package consists of the following clustering algorithms:

* **[Graclus](#graclus)**
* **[VoxelGrid](#voxelgrid)**

All included operations work on varying data types and are implemented both for CPU and GPU.

## Installation

Check that `nvcc` is accessible from terminal, e.g. `nvcc --version`.
If not, add cuda (`/usr/local/cuda/bin`) to your `$PATH`.
Then run:

```
pip install cffi torch-cluster
```

## Graclus

A greedy clustering algorithm of picking an unmarked vertex and matching it with one its unmarked neighbors (that maximizes its edge weight).

```python
import torch
from torch_cluster import graclus_cluster

row = torch.LongTensor([0, 1, 1, 2])
col = torch.LongTensor([1, 0, 2, 1])
weight = torch.Tensor([1, 1, 1, 1])  # Optional edge weights.

cluster = graclus_cluster(row, col, weight)
```

```
print(cluster)
 0  0  1
[torch.LongTensor of size 3]
```

## VoxelGrid

A clustering algorithm, which overlays a regular grid of user-defined size over a point cloud and clusters all points within a voxel.

```python
import torch
from torch_cluster import grid_cluster

pos = torch.Tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
size = torch.Tensor([5, 5])

cluster = grid_cluster(pos, size)
```

```
print(cluster)
 0  5  3  0  1
[torch.LongTensor of size 5]
```

## Running tests

```
python setup.py test
```
