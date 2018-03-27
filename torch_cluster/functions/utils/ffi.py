from ..._ext import ffi


def get_func(name, tensor):
    cuda = '_cuda' if tensor.is_cuda else ''
    return getattr(ffi, 'cluster_{}{}'.format(name, cuda))


def get_dynamic_func(name, tensor):
    typename = type(tensor).__name__.replace('Tensor', '')
    cuda = 'cuda_' if tensor.is_cuda else ''
    return getattr(ffi, 'cluster_{}_{}{}'.format(name, cuda, typename))
