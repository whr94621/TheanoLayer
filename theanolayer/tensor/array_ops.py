# author: Hao-ran Wei
# e-mail: whr94621@gmail.com

from theano import tensor as T


__all__ = [
    'expand_dim',
    'sequence_mask',
    'split',
    'tile_batch'
]

def expand_dim(x, axis=0):

    # x_shp = x.shape

    if axis == -1:
        axis = x.ndim

    new_shp = list(range(axis)) + ['x'] + list(range(axis,x.ndim))

    return x.dimshuffle(*new_shp)


def sequence_mask(lengths, maxlen=None, dtype="float32"):
    """
    :param lengths: Lengths ops, with shape [..., ]

    """

    if maxlen is None:
        maxlen = T.max(lengths)

    row_vector = T.arange(maxlen, dtype=dtype)

    mask = T.switch(expand_dim(lengths, -1) > row_vector,
                    1.0, 0.0)

    return mask


def split(x, axis, split_size):
    """
    Split a ops along one axis.

    Parameters
    ----------
    x: ops

    axis: int

    split_size: int or list/tuple of int
        x.shape[axis] must be divided by split_size.
        or x.shape[axis] is the summation of split_size.
    Returns
    -------
    A tuple of subtensor.
    """
    assert axis < x.ndim, 'Dimension out of range!'

    if isinstance(split_size, int):
        _split_size = [x.shape[axis] // split_size] * split_size

    elif isinstance(split_size, (list, tuple)):
        _split_size = split_size
    else:
        raise TypeError

    if x.ndim == 0:

        return [x for _ in range(len(_split_size))]

    return T.split(x, splits_size=_split_size, n_splits=len(_split_size), axis=axis)

def tile_batch(x, multiplier, batch_dim):

    # out_shp = x.shape[:batch_dim] +_ (x.shape[batch_dim] * multiplier, ) + x.shape

    out_shp = T.concatenate([x.shape[:batch_dim], (x.shape[batch_dim] * multiplier,), x.shape[batch_dim+1:]], axis=0)

    x_tiled = expand_dim(x, axis=batch_dim + 1)

    reps = [1 for _ in range(x_tiled.ndim)]
    reps[batch_dim+1] = multiplier

    x_tiled = T.tile(x_tiled, reps=reps)
    x_tiled = x_tiled.reshape(out_shp, ndim=x.ndim)

    return x_tiled

def _padded_shape(tensor, left_pad, right_pad, axis):

    axis_shp = tensor.shape[axis]


def pad(tensor, paddings, mode="CONSTANT", constant_values=0):
    """
    :type mode: str
    """
    mode = mode.upper()

    assert paddings.ndim == tensor.ndim

    if mode == "CONSTANT":

        pass
    else:
        raise ValueError("Now pad only support constant padding")