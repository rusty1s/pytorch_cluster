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

* **[Graclus](#graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007)
* **[Voxel Grid Pooling](#voxelgrid)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)

All included operations work on varying data types and are implemented both for CPU and GPU.

## Installation

Ensure that at least PyTorch 0.4.1 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 0.4.1

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```

Then run:

```
pip install torch-cluster
```

If you are running into any installation problems, please create an [issue](https://github.com/rusty1s/pytorch_cluster/issues).
Be sure to import `torch` first before using this package to resolve symbols the dynamic linker must see.

## Graclus

A greedy clustering algorithm of picking an unmarked vertex and matching it with one its unmarked neighbors (that maximizes its edge weight).
The GPU algorithm is adapted from Fagginger Auer and Bisseling: [A GPU Algorithm for Greedy Graph Matching](http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf) (LNCS 2012)

```python
import torch
from torch_cluster import graclus_cluster

row = torch.tensor([0, 1, 1, 2])
col = torch.tensor([1, 0, 2, 1])
weight = torch.Tensor([1, 1, 1, 1])  # Optional edge weights.

cluster = graclus_cluster(row, col, weight)
```

```
print(cluster)
tensor([0, 0, 1])
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
tensor([0, 5, 3, 0, 1])
```

## Running tests

```
python setup.py test
```
