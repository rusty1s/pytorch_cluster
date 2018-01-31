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

This package consists of a small extension library of highly optimised graph cluster algorithms for the use in [PyTorch](http://pytorch.org/).
All included operations work on varying data types and are implemented both for CPU and GPU.

## Installation

Check that `nvcc` is accessible from terminal, e.g. `nvcc --version`.
If not, add cuda (`/usr/local/cuda/bin`) to your `$PATH`.
Then run:

```
pip install cffi torch-cluster
python setup.py install
```

## Running tests

```
python setup.py test
```
