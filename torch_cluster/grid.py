from .utils.ffi import grid


def grid_cluster(pos, size, batch=None, start=None, end=None):
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    grid(None, None, None, None)
