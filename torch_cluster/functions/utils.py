from .._ext import ffi


def get_func(name, tensor):
    typename = type(tensor).__name__.replace('Tensor', '')
    cuda = 'cuda_' if tensor.is_cuda else ''
    func = getattr(ffi, 'cluster_{}_{}{}'.format(name, cuda, typename))
    return func
