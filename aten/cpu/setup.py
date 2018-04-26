from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='cluster',
    ext_modules=[CppExtension('cluster_cpu', ['cluster.cpp'])],
    cmdclass={'build_ext': BuildExtension},
)
