[pypi-image]: https://badge.fury.io/py/torch-cluster.svg
[pypi-url]: https://pypi.python.org/pypi/torch-cluster
[testing-image]: https://github.com/rusty1s/pytorch_cluster/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/rusty1s/pytorch_cluster/actions/workflows/testing.yml
[linting-image]: https://github.com/rusty1s/pytorch_cluster/actions/workflows/linting.yml/badge.svg
[linting-url]: https://github.com/rusty1s/pytorch_cluster/actions/workflows/linting.yml
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_cluster/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_cluster?branch=master

# PyTorch Cluster

[![PyPI Version][pypi-image]][pypi-url]
[![Testing Status][testing-image]][testing-url]
[![Linting Status][linting-image]][linting-url]
[![Code Coverage][coverage-image]][coverage-url]

--------------------------------------------------------------------------------

This package consists of a small extension library of highly optimized graph cluster algorithms for the use in [PyTorch](http://pytorch.org/).
The package consists of the following clustering algorithms:

* **[Graclus](#graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007)
* **[Voxel Grid Pooling](#voxelgrid)** from, *e.g.*, Simonovsky and Komodakis: [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) (CVPR 2017)
* **[Iterative Farthest Point Sampling](#farthestpointsampling)** from, *e.g.* Qi *et al.*: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) (NIPS 2017)
* **[k-NN](#knn-graph)** and **[Radius](#radius-graph)** graph generation
* Clustering based on **[Nearest](#nearest)** points
* **[Random Walk Sampling](#randomwalk-sampling)** from, *e.g.*, Grover and Leskovec: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (KDD 2016)

All included operations work on varying data types and are implemented both for CPU and GPU.

## Installation

### Anaconda

**Update:** You can now install `pytorch-cluster` via [Anaconda](https://anaconda.org/pyg/pytorch-cluster) for all major OS/PyTorch/CUDA combinations 🤗
Given that you have [`pytorch >= 1.8.0` installed](https://pytorch.org/get-started/locally/), simply run

```
conda install pytorch-cluster -c pyg
```

### Binaries

We alternatively provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://data.pyg.org/whl).

#### PyTorch 1.11

To install the binaries for PyTorch 1.11.0, simply run

```
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation.

|             | `cpu` | `cu102` | `cu113` | `cu115` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |
| **Windows** | ✅    |         | ✅      | ✅      |
| **macOS**   | ✅    |         |         |         |

#### PyTorch 1.10

To install the binaries for PyTorch 1.10.0, PyTorch 1.10.1 and PyTorch 1.10.2, simply run

```
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu111`, or `cu113` depending on your PyTorch installation.

|             | `cpu` | `cu102` | `cu111` | `cu113` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      | ✅      |
| **macOS**   | ✅    |         |         |         |

**Note:** Binaries of older versions are also provided for PyTorch 1.4.0, PyTorch 1.5.0, PyTorch 1.6.0, PyTorch 1.7.0/1.7.1, PyTorch 1.8.0/1.8.1 and PyTorch 1.9.0 (following the same procedure).
For older versions, you might need to explicitly specify the latest supported version number in order to prevent a manual installation from source.
You can look up the latest supported version number [here](https://data.pyg.org/whl).

### From source

Ensure that at least PyTorch 1.4.0 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 1.4.0

$ python -c "import torch; print(torch.__version__)"
>>> 1.1.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```

Then run:

```
pip install torch-cluster
```

When running in a docker container without NVIDIA driver, PyTorch needs to evaluate the compute capabilities and may fail.
In this case, ensure that the compute capabilities are set via `TORCH_CUDA_ARCH_LIST`, *e.g.*:

```
export TORCH_CUDA_ARCH_LIST = "6.0 6.1 7.2+PTX 7.5+PTX"
```

## Functions

### Graclus

A greedy clustering algorithm of picking an unmarked vertex and matching it with one its unmarked neighbors (that maximizes its edge weight).
The GPU algorithm is adapted from Fagginger Auer and Bisseling: [A GPU Algorithm for Greedy Graph Matching](http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf) (LNCS 2012)

```python
import torch
from torch_cluster import graclus_cluster

row = torch.tensor([0, 1, 1, 2])
col = torch.tensor([1, 0, 2, 1])
weight = torch.tensor([1., 1., 1., 1.])  # Optional edge weights.

cluster = graclus_cluster(row, col, weight)
```

```
print(cluster)
tensor([0, 0, 1])
```

### VoxelGrid

A clustering algorithm, which overlays a regular grid of user-defined size over a point cloud and clusters all points within a voxel.

```python
import torch
from torch_cluster import grid_cluster

pos = torch.tensor([[0., 0.], [11., 9.], [2., 8.], [2., 2.], [8., 3.]])
size = torch.Tensor([5, 5])

cluster = grid_cluster(pos, size)
```

```
print(cluster)
tensor([0, 5, 3, 0, 1])
```

### FarthestPointSampling

A sampling algorithm, which iteratively samples the most distant point with regard to the rest points.

```python
import torch
from torch_cluster import fps

x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
batch = torch.tensor([0, 0, 0, 0])
index = fps(x, batch, ratio=0.5, random_start=False)
```

```
print(index)
tensor([0, 3])
```

### kNN-Graph

Computes graph edges to the nearest *k* points.

**Args:**

* **x** *(Tensor)*: Node feature matrix of shape `[N, F]`.
* **k** *(int)*: The number of neighbors.
* **batch** *(LongTensor, optional)*: Batch vector of shape `[N]`, which assigns each node to a specific example. `batch` needs to be sorted. (default: `None`)
* **loop** *(bool, optional)*: If `True`, the graph will contain self-loops. (default: `False`)
* **flow** *(string, optional)*: The flow direction when using in combination with message passing (`"source_to_target"` or `"target_to_source"`). (default: `"source_to_target"`)
* **cosine** *(boolean, optional)*: If `True`, will use the Cosine distance instead of Euclidean distance to find nearest neighbors. (default: `False`)
* **num_workers** *(int)*: Number of workers to use for computation. Has no effect in case `batch` is not `None`, or the input lies on the GPU. (default: `1`)

```python
import torch
from torch_cluster import knn_graph

x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
batch = torch.tensor([0, 0, 0, 0])
edge_index = knn_graph(x, k=2, batch=batch, loop=False)
```

```
print(edge_index)
tensor([[1, 2, 0, 3, 0, 3, 1, 2],
        [0, 0, 1, 1, 2, 2, 3, 3]])
```

### Radius-Graph

Computes graph edges to all points within a given distance.

**Args:**

* **x** *(Tensor)*: Node feature matrix of shape `[N, F]`.
* **r** *(float)*: The radius.
* **batch** *(LongTensor, optional)*: Batch vector of shape `[N]`, which assigns each node to a specific example. `batch` needs to be sorted. (default: `None`)
* **loop** *(bool, optional)*: If `True`, the graph will contain self-loops. (default: `False`)
* **max_num_neighbors** *(int, optional)*: The maximum number of neighbors to return for each element. If the number of actual neighbors is greater than `max_num_neighbors`, returned neighbors are picked randomly. (default: `32`)
* **flow** *(string, optional)*: The flow direction when using in combination with message passing (`"source_to_target"` or `"target_to_source"`). (default: `"source_to_target"`)
* **num_workers** *(int)*: Number of workers to use for computation. Has no effect in case `batch` is not `None`, or the input lies on the GPU. (default: `1`)

```python
import torch
from torch_cluster import radius_graph

x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
batch = torch.tensor([0, 0, 0, 0])
edge_index = radius_graph(x, r=2.5, batch=batch, loop=False)
```

```
print(edge_index)
tensor([[1, 2, 0, 3, 0, 3, 1, 2],
        [0, 0, 1, 1, 2, 2, 3, 3]])
```

### Nearest

Clusters points in *x* together which are nearest to a given query point in *y*.
`batch_{x,y}` vectors need to be sorted.

```python
import torch
from torch_cluster import nearest

x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
batch_x = torch.tensor([0, 0, 0, 0])
y = torch.Tensor([[-1, 0], [1, 0]])
batch_y = torch.tensor([0, 0])
cluster = nearest(x, y, batch_x, batch_y)
```

```
print(cluster)
tensor([0, 0, 1, 1])
```

### RandomWalk-Sampling

Samples random walks of length `walk_length` from all node indices in `start` in the graph given by `(row, col)`.

```python
import torch
from torch_cluster import random_walk

row = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
col = torch.tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3])
start = torch.tensor([0, 1, 2, 3, 4])

walk = random_walk(row, col, start, walk_length=3)
```

```
print(walk)
tensor([[0, 1, 2, 4],
        [1, 3, 4, 2],
        [2, 4, 2, 1],
        [3, 4, 2, 4],
        [4, 3, 1, 0]])
```

## Running tests

```
pytest
```

## C++ API

`torch-cluster` also offers a C++ API that contains C++ equivalent of python models.

```
mkdir build
cd build
# Add -DWITH_CUDA=on support for the CUDA if needed
cmake ..
make
make install
```
