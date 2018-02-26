from os import path as osp
from itertools import product

import pytest
import json
import torch
from torch_cluster import sparse_grid_cluster

from .utils import tensors, Tensor

f = open(osp.join(osp.dirname(__file__), 'sparse_grid.json'), 'r')
data = json.load(f)
f.close()


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_sparse_grid_cluster_cpu(tensor, i):
    position = Tensor(tensor, data[i]['position'])
    size = torch.LongTensor(data[i]['size'])
    batch = data[i].get('batch')
    start = data[i].get('start')
    expected = torch.LongTensor(data[i]['expected'])

    if batch is None:
        output = sparse_grid_cluster(position, size, batch, start)
        assert output.tolist() == expected.tolist()
    else:
        batch = torch.LongTensor(batch)
        expected_batch = torch.LongTensor(data[i]['expected_batch'])
        output = sparse_grid_cluster(position, size, batch, start)
        assert output[0].tolist() == expected.tolist()
        assert output[1].tolist() == expected_batch.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_sparse_grid_cluster_gpu(tensor, i):  # pragma: no cover
    position = Tensor(tensor, data[i]['position']).cuda()
    size = torch.cuda.LongTensor(data[i]['size'])
    batch = data[i].get('batch')
    start = data[i].get('start')
    expected = torch.LongTensor(data[i]['expected'])

    if batch is None:
        output = sparse_grid_cluster(position, size, batch, start)
        assert output.cpu().tolist() == expected.tolist()
    else:
        batch = torch.cuda.LongTensor(batch)
        expected_batch = torch.LongTensor(data[i]['expected_batch'])
        output = sparse_grid_cluster(position, size, batch, start)
        assert output[0].cpu().tolist() == expected.tolist()
        assert output[1].cpu().tolist() == expected_batch.tolist()
