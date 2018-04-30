from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cluster_cuda',
    ext_modules=[
        CUDAExtension('cluster_cuda', ['cluster.cpp', 'cluster_kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension},
)
