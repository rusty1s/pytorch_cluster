from itertools import product

import pytest
import torch
from torch_cluster import knn, knn_graph
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
    x = torch.tensor([[-1.0320,  0.2380,  0.2380],
                      [-1.3050, -0.0930,  0.6420],
                      [-0.3190, -0.0410,  1.2150],
                      [1.1400, -0.5390, -0.3140],
                      [0.8410,  0.8290,  0.6090],
                      [-1.4380, -0.2420, -0.3260],
                      [-2.2980,  0.7160,  0.9320],
                      [-1.3680, -0.4390,  0.1380],
                      [-0.6710,  0.6060,  1.1800],
                      [0.3950, -0.0790,  1.4920]],).to(device)
    k = 3
    truth = set({(4, 8), (2, 8), (9, 8), (8, 0), (0, 7), (2, 1), (9, 4),
                 (5, 1), (4, 9), (2, 9), (8, 1), (1, 5), (5, 0), (3, 2),
                 (8, 2), (7, 1), (6, 0), (3, 9), (0, 5), (7, 5), (4, 2),
                 (1, 0), (0, 1), (7, 0), (6, 8), (9, 2), (6, 1), (5, 7),
                 (1, 7), (3, 4)})

    row, col = knn_graph(x, k=k, flow='target_to_source',
                         batch=None, n_threads=24, loop=False)

    edges = set([(i, j) for (i, j) in zip(list(row.cpu().numpy()),
                                          list(col.cpu().numpy()))])

    assert(truth == edges)

    row, col = knn_graph(x, k=k, flow='target_to_source',
                         batch=None, n_threads=12, loop=False)

    edges = set([(i, j) for (i, j) in zip(list(row.cpu().numpy()),
                                          list(col.cpu().numpy()))])

    assert(truth == edges)

    row, col = knn_graph(x, k=k, flow='target_to_source',
                         batch=None, n_threads=1, loop=False)

    edges = set([(i, j) for (i, j) in zip(list(row.cpu().numpy()),
                                          list(col.cpu().numpy()))])

    assert(truth == edges)
