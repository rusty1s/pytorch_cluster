from os import path as osp
from itertools import product

import pytest
import json
import torch
from torch_cluster import dense_grid_cluster

from .utils import tensors, Tensor

f = open(osp.join(osp.dirname(__file__), 'dense_grid.json'), 'r')
data = json.load(f)
f.close()


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_dense_grid_cluster_cpu(tensor, i):
    position = Tensor(tensor, data[i]['position'])
    size = torch.LongTensor(data[i]['size'])
    batch = data[i].get('batch')
    batch = None if batch is None else torch.LongTensor(batch)
    start = data[i].get('start')
    end = data[i].get('end')
    expected = torch.LongTensor(data[i]['expected'])
    expected_C = data[i]['expected_C']

    output = dense_grid_cluster(position, size, batch, start, end)
    assert output[0].tolist() == expected.tolist()
    assert output[1] == expected_C


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_dense_grid_cluster_gpu(tensor, i):  # pragma: no cover
    position = Tensor(tensor, data[i]['position']).cuda()
    size = torch.cuda.LongTensor(data[i]['size'])
    batch = data[i].get('batch')
    batch = None if batch is None else torch.cuda.LongTensor(batch)
    start = data[i].get('start')
    end = data[i].get('end')
    expected = torch.LongTensor(data[i]['expected'])
    expected_C = data[i]['expected_C']

    output = dense_grid_cluster(position, size, batch, start, end)
    assert output[0].cpu().tolist() == expected.tolist()
    assert output[1] == expected_C
