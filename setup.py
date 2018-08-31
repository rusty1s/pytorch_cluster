from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension

ext_modules = [
    CppExtension('graclus_cpu', ['cpu/graclus.cpp']),
    CppExtension('grid_cpu', ['cpu/grid.cpp']),
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if torch.cuda.is_available():
    ext_modules += [
        CUDAExtension('graclus_cuda',
                      ['cuda/graclus.cpp', 'cuda/graclus_kernel.cu']),
        CUDAExtension('grid_cuda', ['cuda/grid.cpp', 'cuda/grid_kernel.cu']),
    ]

__version__ = '1.1.5'
url = 'https://github.com/rusty1s/pytorch_cluster'

install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_cluster',
    version=__version__,
    description='PyTorch Extension Library of Optimized Graph Cluster '
    'Algorithms',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'cluster', 'geometric-deep-learning', 'graph'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
