import numpy as np
import theano.tensor as T
from .base import Module

__all__ = [
    'LayerNorm'
]

class LayerNorm(Module):

    scale_add = 0.0
    scale_mul = 1.0
    _eps = np.float32(1e-5)

    def __init__(self,
                 input_size,
                 parameters,
                 prefix="layer_norm"):

        super(LayerNorm, self).__init__(parameters=parameters, prefix=prefix)

        self._register_params(name="lnb",
                              value_or_shape=LayerNorm.scale_add * np.ones((input_size, )).astype(np.float32))

        self._register_params(name="lns",
                              value_or_shape=LayerNorm.scale_mul * np.ones((input_size, )).astype(np.float32))

    def forward(self, inputs, *args, **kwargs):

        outs = (inputs - T.mean(inputs, axis=-1, keepdims=True)) / T.sqrt(T.var(inputs, axis=-1, keepdims=True) + LayerNorm._eps)
        outs = self._get_shared("lns") * outs + self._get_shared("lnb")

        return outs