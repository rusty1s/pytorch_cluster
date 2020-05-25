from itertools import product

import pytest
import torch
from torch_cluster import knn, knn_graph
import pickle
from .utils import grad_dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)
    y = tensor([
        [1, 0],
        [-1, 0],
    ], dtype, device)

    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 1], torch.long, device)

    row, col = knn(x, y, 2, batch_x, batch_y)
    col = col.view(-1, 2).sort(dim=-1)[0].view(-1)

    assert row.tolist() == [0, 0, 1, 1]
    assert col.tolist() == [2, 3, 4, 5]

    if x.is_cuda:
        row, col = knn(x, y, 2, batch_x, batch_y, cosine=True)
        assert row.tolist() == [0, 0, 1, 1]
        assert col.tolist() == [0, 1, 4, 5]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    row, col = knn_graph(x, k=2, flow='target_to_source')
    col = col.view(-1, 2).sort(dim=-1)[0].view(-1)
    assert row.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert col.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]

    row, col = knn_graph(x, k=2, flow='source_to_target')
    row = row.view(-1, 2).sort(dim=-1)[0].view(-1)
    assert row.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]
    assert col.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_graph_large(dtype, device):
    d = pickle.load(open("test/knn_test_large.pkl", "rb"))
    x = d['x'].to(device)
    k = d['k']
    truth = d['edges']

    row, col = knn_graph(x, k=k, flow='source_to_target',
                         batch=None, n_threads=24)

    edges = set([(i, j) for (i, j) in zip(list(row.cpu().numpy()),
                                          list(col.cpu().numpy()))])

    assert(truth == edges)

    row, col = knn_graph(x, k=k, flow='source_to_target',
                         batch=None, n_threads=12)

    edges = set([(i, j) for (i, j) in zip(list(row.cpu().numpy()),
                                          list(col.cpu().numpy()))])

    assert(truth == edges)

    row, col = knn_graph(x, k=k, flow='source_to_target',
                         batch=None, n_threads=1)

    edges = set([(i, j) for (i, j) in zip(list(row.cpu().numpy()),
                                          list(col.cpu().numpy()))])

    assert(truth == edges)
