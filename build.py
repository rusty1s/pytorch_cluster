import os.path as osp
import subprocess

import torch
from torch.utils.ffi import create_extension

files = ['Graclus', 'Grid']

headers = ['aten/TH/TH{}.h'.format(f) for f in files]
sources = ['aten/TH/TH{}.c'.format(f) for f in files]
include_dirs = ['aten/TH']
define_macros = []
extra_objects = []
extra_compile_args = ['-std=c99']
with_cuda = False

if torch.cuda.is_available():
    subprocess.call(['./build.sh', osp.dirname(torch.__file__)])

    headers += ['aten/THCC/THCC{}.h'.format(f) for f in files]
    sources += ['aten/THCC/THCC{}.c'.format(f) for f in files]
    include_dirs += ['aten/THCC']
    define_macros += [('WITH_CUDA', None)]
    extra_objects += ['torch_cluster/_ext/THC.so']
    with_cuda = True

ffi = create_extension(
    name='torch_cluster._ext.ffi',
    package=True,
    headers=headers,
    sources=sources,
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
    with_cuda=with_cuda,
    relative_to=__file__)

if __name__ == '__main__':
    ffi.build()
