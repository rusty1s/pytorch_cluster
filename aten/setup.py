import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension

ext_modules = [CppExtension(name='cluster_cpu', sources=['cpu/cluster.cpp'])]

if torch.cuda.is_available():
    ext_modules += [
        CUDAExtension(
            name='cluster_cuda',
            sources=['cuda/cluster.cpp', 'cuda/cluster_kernel.cu'])
    ]

setup(
    name='cluster',
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
)
