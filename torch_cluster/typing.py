import torch

WITH_PTR_LIST = hasattr(torch.ops.torch_cluster, 'fp_ptr_list')
print(WITH_PTR_LIST)
