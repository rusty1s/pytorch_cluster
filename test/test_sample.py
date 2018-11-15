import pytest
import torch
import numpy as np
from torch_geometric.data import Batch
from numpy.testing import assert_almost_equal

from capsules.utils.sample import sample_farthest, batch_slices, radius_query_edges

from .utils import tensor, grad_dtypes, devices

@pytest.mark.parametrize('device', devices)
def test_batch_slices(device):
    # test sample case for correctness
    batch = tensor([0] * 100 + [1] * 50 + [2] * 42, dtype=torch.long, device=device)

    slices, sizes = batch_slices(batch, sizes=True)
    slices, sizes = slices.cpu().tolist(), sizes.cpu().tolist()

    assert slices == [0, 99, 100, 149, 150, 191]
    assert sizes == [100, 50, 42]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('dtype', grad_dtypes)
def test_fps(dtype):
    # test simple case for correctness
    batch = [0] * 10
    points = [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]
    random_points = np.random.uniform(-1, 1, size=(6, 3))
    random_points[:, 2] = 0
    random_points = random_points.tolist()

    batch = tensor(batch, dtype=torch.long, device='cuda')
    pos = tensor(points + random_points, dtype=dtype, device='cuda')

    idx = sample_farthest(batch, pos, num_sampled=4, index=True)

    # needs update since isin is missing (sort indices, then compare?)
    # assert isin(idx, tensor([0, 1, 2, 3], dtype=torch.long, device='cuda'), False).all().cpu().item() == 1

    # test variable number of points for each element in a batch
    batch = [0] * 100 + [1] * 50
    points1 = np.random.uniform(-1, 1, size=(100, 3)).tolist()
    points2 = np.random.uniform(-1, 1, size=(50, 3)).tolist()

    batch = tensor(batch, dtype=torch.long, device='cuda')
    pos = tensor(points1 + points2, dtype=dtype, device='cuda')

    mask = sample_farthest(batch, pos, num_sampled=75, index=False)
    assert mask[batch == 0].sum().item() == 75
    assert mask[batch == 1].sum().item() == 50


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('dtype', grad_dtypes)
def test_radius_edges(dtype):
    batch = [0] * 100 + [1] * 50 + [2] * 42
    points = np.random.uniform(-1, 1, size=(192, 3))

    query_batch = [0] * 10 + [1] * 15 + [2] * 20
    query_points = np.random.uniform(-1, 1, size=(45, 3))

    radius = 0.5

    batch = tensor(batch, dtype=torch.long, device='cuda')
    query_batch = tensor(query_batch, dtype=torch.long, device='cuda')
    pos = tensor(points, dtype=dtype, device='cuda')
    query_pos = tensor(query_points, dtype=dtype, device='cuda')

    edge_index = radius_query_edges(batch, pos, query_batch, query_pos, radius=radius, max_num_neighbors=128)
    row, col = edge_index
    dist = torch.norm(pos[col] - query_pos[row], p=2, dim=1)
    assert (dist <= radius).all().item()
