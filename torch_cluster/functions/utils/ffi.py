from ..._ext import ffi


def _get_func(name, tensor):
    cuda = '_cuda' if tensor.is_cuda else ''
    return getattr(ffi, 'cluster_{}{}'.format(name, cuda))


def _get_typed_func(name, tensor):
    typename = type(tensor).__name__.replace('Tensor', '')
    cuda = 'cuda_' if tensor.is_cuda else ''
    return getattr(ffi, 'cluster_{}_{}{}'.format(name, cuda, typename))


def ffi_serial(row, col, degree, weight=None):
    output = row.new(degree.size(0)).fill_(-1)
    if weight is None:
        func = _get_func('serial', row)
        func(output, row, col, degree)
        return output
    else:
        func = _get_typed_func('serial', weight)
        func(output, row, col, degree, weight)
        return output


def ffi_grid(position, size, count):
    C = count.prod()
    output = count.new(position.size(0), 1)
    func = _get_typed_func('grid', position)
    func(C, output, position, size, count)
    output = output.squeeze(-1)
    return output
