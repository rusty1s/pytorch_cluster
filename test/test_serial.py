# import torch
# from torch_cluster import serial_cluster


def test_serial():
    pass
    # ed_index = torch.LongTensor([[0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6],
    #                              [2, 3, 6, 5, 0, 0, 4, 5, 3, 1, 3, 6, 0, 3]])
    # edge_attr = torch.Tensor([2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2])
    # rid = torch.arange(edge_index.max() + 1, out=edge_index.new())
    # output = random_cluster(edge_index, rid=rid, perm_edges=False)

    # expected_output = [0, 1, 2, 0, 3, 1, 4]
    # assert output.tolist() == expected_output

    # TODO: Test only if conditions are met:
    # * at most two pairs with the same cluster
    # * pairs need to be neighbors of each other
    # TODO: Rename to serial
