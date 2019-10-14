from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

extra_compile_args = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    extra_compile_args += ['-DVERSION_GE_1_3']

ext_modules = [
    CppExtension('torch_cluster.graclus_cpu', ['cpu/graclus.cpp'],
                 extra_compile_args=extra_compile_args),
    CppExtension('torch_cluster.grid_cpu', ['cpu/grid.cpp']),
    CppExtension('torch_cluster.fps_cpu', ['cpu/fps.cpp'],
                 extra_compile_args=extra_compile_args),
    CppExtension('torch_cluster.rw_cpu', ['cpu/rw.cpp'],
                 extra_compile_args=extra_compile_args),
    CppExtension('torch_cluster.sampler_cpu', ['cpu/sampler.cpp'],
                 extra_compile_args=extra_compile_args),
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if CUDA_HOME is not None:
    ext_modules += [
        CUDAExtension('torch_cluster.graclus_cuda',
                      ['cuda/graclus.cpp', 'cuda/graclus_kernel.cu'],
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_cluster.grid_cuda',
                      ['cuda/grid.cpp', 'cuda/grid_kernel.cu'],
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_cluster.fps_cuda',
                      ['cuda/fps.cpp', 'cuda/fps_kernel.cu'],
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_cluster.nearest_cuda',
                      ['cuda/nearest.cpp', 'cuda/nearest_kernel.cu'],
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_cluster.knn_cuda',
                      ['cuda/knn.cpp', 'cuda/knn_kernel.cu'],
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_cluster.radius_cuda',
                      ['cuda/radius.cpp', 'cuda/radius_kernel.cu'],
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_cluster.rw_cuda',
                      ['cuda/rw.cpp', 'cuda/rw_kernel.cu'],
                      extra_compile_args=extra_compile_args),
    ]

__version__ = '1.4.5'
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
