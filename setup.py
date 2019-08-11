from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

ext_modules = [
    CppExtension('torch_cluster.graclus_cpu', ['cpu/graclus.cpp']),
    CppExtension('torch_cluster.grid_cpu', ['cpu/grid.cpp']),
    CppExtension('torch_cluster.fps_cpu', ['cpu/fps.cpp']),
    CppExtension('torch_cluster.rw_cpu', ['cpu/rw.cpp']),
    CppExtension('torch_cluster.sampler_cpu', ['cpu/sampler.cpp']),
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if CUDA_HOME is not None:
    ext_modules += [
        CUDAExtension('torch_cluster.graclus_cuda',
                      ['cuda/graclus.cpp', 'cuda/graclus_kernel.cu']),
        CUDAExtension('torch_cluster.grid_cuda',
                      ['cuda/grid.cpp', 'cuda/grid_kernel.cu']),
        CUDAExtension('torch_cluster.fps_cuda',
                      ['cuda/fps.cpp', 'cuda/fps_kernel.cu']),
        CUDAExtension('torch_cluster.nearest_cuda',
                      ['cuda/nearest.cpp', 'cuda/nearest_kernel.cu']),
        CUDAExtension('torch_cluster.knn_cuda',
                      ['cuda/knn.cpp', 'cuda/knn_kernel.cu']),
        CUDAExtension('torch_cluster.radius_cuda',
                      ['cuda/radius.cpp', 'cuda/radius_kernel.cu']),
        CUDAExtension('torch_cluster.rw_cuda',
                      ['cuda/rw.cpp', 'cuda/rw_kernel.cu']),
    ]

__version__ = '1.4.4'
url = 'https://github.com/rusty1s/pytorch_cluster'

install_requires = ['scipy']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_cluster',
    version=__version__,
    description=('PyTorch Extension Library of Optimized Graph Cluster '
                 'Algorithms'),
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'cluster-algorithms',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
