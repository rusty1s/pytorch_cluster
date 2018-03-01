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
    batch = None if batch is None else torch.LongTensor(batch)
    start = data[i].get('start')
    start = None if start is None else torch.LongTensor(start)
    expected = torch.LongTensor(data[i]['expected'])

    output = sparse_grid_cluster(position, size, batch, start)

    if batch is None:
        assert output.tolist() == expected.tolist()
    else:
        expected_batch = torch.LongTensor(data[i]['expected_batch'])
        assert output[0].tolist() == expected.tolist()
        assert output[1].tolist() == expected_batch.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_sparse_grid_cluster_gpu(tensor, i):  # pragma: no cover
    position = Tensor(tensor, data[i]['position']).cuda()
    size = torch.cuda.LongTensor(data[i]['size'])
    batch = data[i].get('batch')
    batch = None if batch is None else torch.cuda.LongTensor(batch)
    start = data[i].get('start')
    start = None if start is None else torch.cuda.LongTensor(start)
    expected = torch.LongTensor(data[i]['expected'])

    output = sparse_grid_cluster(position, size, batch, start)

    if batch is None:
        assert output.cpu().tolist() == expected.tolist()
    else:
        expected_batch = torch.LongTensor(data[i]['expected_batch'])
        assert output[0].cpu().tolist() == expected.tolist()
        assert output[1].cpu().tolist() == expected_batch.tolist()
