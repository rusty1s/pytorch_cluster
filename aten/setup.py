import glob
from setuptools import setup

import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension

ext_modules = [CppExtension('cluster_cpu', ['cpu/cluster.cpp'])]

if torch.cuda.is_available():
    ext_modules += [
        CUDAExtension('cluster_cuda',
                      ['cuda/cluster.cpp'] + glob.glob('cuda/*.cu'))
    ]

setup(
    name='cluster',
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
)
