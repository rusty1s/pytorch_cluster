cpu_tensors = [
    'ByteTensor', 'CharTensor', 'ShortTensor', 'IntTensor', 'LongTensor',
    'FloatTensor', 'DoubleTensor'
]

gpu_tensors = ['cuda.{}'.format(t) for t in cpu_tensors + ['HalfTensor']]
