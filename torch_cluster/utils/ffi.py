from .._ext import ffi


def get_func(name, is_cuda, tensor=None):
    prefix = 'THCC' if is_cuda else 'TH'
    prefix += 'Tensor' if tensor is None else tensor.type().split('.')[-1]
    return getattr(ffi, '{}_{}'.format(prefix, name))


def graclus(self, row, col, weight=None):
    func = get_func('graclus', self.is_cuda, weight)
    func(self, row, col) if weight is None else func(self, row, col, weight)


def grid(self, pos, size, count):
    func = get_func('grid', self.is_cuda, pos)
    func(self, pos, size, count)
