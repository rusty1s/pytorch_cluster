# PyTorch Cluster

--------------------------------------------------------------------------------

This package consists of a small extension library of highly optimised graph cluster algorithms for the use in [PyTorch](http://pytorch.org/).
All included operations work on varying data types and are implemented both for CPU and GPU.

## Installation

Check that `nvcc` is accessible from terminal, e.g. `nvcc --version`.
If not, add cuda (`/usr/local/cuda/bin`) to your `$PATH`.
Then run:

```
pip install cffi
python setup.py install
```

## Running tests

```
python setup.py test
```
