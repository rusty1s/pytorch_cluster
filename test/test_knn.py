import math
from itertools import product

import pytest
import torch
import scipy.spatial
from torch_cluster import knn, knn_graph

from .utils import grad_dtypes, devices, tensor


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


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

    edge_index, distances = knn(x, y, 2, return_distances=True)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 0), (1, 1)])
    assert torch.allclose(distances, distances.new_tensor([1.0, 1.0, 1.0, 1.0]))

    edge_index, distances = knn(x, y, 2, batch_x, batch_y, return_distances=True)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])
    assert torch.allclose(distances, distances.new_tensor([1.0, 1.0, 1.0, 1.0]))

    if x.is_cuda:
        edge_index, distances = knn(
            x, y, 2, batch_x, batch_y, cosine=True, return_distances=True
        )
        assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])
        assert torch.allclose(distances, distances.new_tensor(
            [1.0 - math.cos(math.pi / 4.0) for _ in range(4)]
        ))

    # Skipping a batch
    batch_x = tensor([0, 0, 0, 0, 2, 2, 2, 2], torch.long, device)
    batch_y = tensor([0, 2], torch.long, device)
    edge_index,distances = knn(x, y, 2, batch_x, batch_y, return_distances=True)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])
    assert torch.allclose(distances, distances.new_tensor([1.0, 1.0, 1.0, 1.0]))


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_jit(dtype, device):
    @torch.jit.script
    def knn_jit(x: torch.Tensor, y: torch.Tensor, k: int, batch_x: torch.Tensor,
                batch_y: torch.Tensor):
        return knn(x, y, k, batch_x, batch_y)

    @torch.jit.script
    def knn_jit_distance(x: torch.Tensor, y: torch.Tensor, k: int,
                         batch_x: torch.Tensor, batch_y: torch.Tensor):
        return knn(x, y, k, batch_x, batch_y, return_distances=True)

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

    edge_index = knn_jit(x, y, 2, batch_x, batch_y)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])

    edge_index, distances = knn_jit_distance(x, y, 2, batch_x, batch_y)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])
    assert torch.allclose(distances, distances.new_tensor([1.0, 1.0, 1.0, 1.0]))


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    edge_index, distances = knn_graph(
        x, k=2, flow='target_to_source', return_distances=True
    )
    assert to_set(edge_index) == set([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1),
                                      (2, 3), (3, 0), (3, 2)])
    assert torch.allclose(distances, distances.new_tensor([4.0 for _ in range(8)]))

    edge_index = knn_graph(
        x, k=2, flow='source_to_target', return_distances=False
    )
    assert to_set(edge_index) == set([(1, 0), (3, 0), (0, 1), (2, 1), (1, 2),
                                      (3, 2), (0, 3), (2, 3)])


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_graph_jit(dtype, device):
    @torch.jit.script
    def knn_graph_jit(x: torch.Tensor, k: int):
        return knn_graph(x, k, flow="target_to_source")

    @torch.jit.script
    def knn_graph_jit_distance(x: torch.Tensor, k: int):
        return knn_graph(x, k, flow="target_to_source", return_distances=True)

    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    edge_index = knn_graph_jit(x, k=2)
    assert to_set(edge_index) == set([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1),
                                      (2, 3), (3, 0), (3, 2)])

    edge_index, distances = knn_graph_jit_distance(x, k=2)
    assert to_set(edge_index) == set([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1),
                                      (2, 3), (3, 0), (3, 2)])
    assert torch.allclose(distances, distances.new_tensor([4.0 for _ in range(8)]))


@pytest.mark.parametrize('dtype,device', product([torch.float], devices))
def test_knn_graph_large(dtype, device):
    x = torch.randn(1000, 3, dtype=dtype, device=device)

    edge_index, distances = knn_graph(
        x, k=5, flow='target_to_source', loop=True, return_distances=True
    )

    tree = scipy.spatial.cKDTree(x.cpu().numpy())
    dist, col = tree.query(x.cpu(), k=5)
    truth = set([(i, j) for i, ns in enumerate(col) for j in ns])

    assert to_set(edge_index.cpu()) == truth
    assert torch.allclose(
        distances, torch.from_numpy(dist).to(distances).flatten().pow(2)
    )
