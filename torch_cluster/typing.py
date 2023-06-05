import torch

WITH_PTR_LIST = hasattr(torch.ops.torch_cluster, 'fps_ptr_list')
