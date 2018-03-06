import theano.tensor as T
from theanolayer.utils import nest
from .base import Module

__all__ = [
    'Dense'
]

class Dense(Module):

    def __init__(self,
                 input_size,
                 num_units,
                 parameters,
                 use_bias=True,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 prefix="dense"):

        super(Dense, self).__init__(parameters, prefix)

        input_size = nest.flatten(input_size)

        self.input_size = sum(input_size)
        self.num_units = num_units
        self.activation = activation
        self.use_bias = use_bias

        self._register_params(name="W", value_or_shape=(self.input_size, self.num_units),
                              initializer=kernel_initializer)

        if use_bias is True:
            self._register_params(name="b", value_or_shape=(self.num_units,),
                                  initializer=bias_initializer)

    def forward(self, inputs, *args, **kwargs):

        inputs = nest.flatten(inputs)

        ndim0 = inputs[0].ndim

        for t in inputs[1:]:
            if t.ndim != ndim0:
                raise ValueError("Inputs should all have {ndim} rank(s).".format(ndim=ndim0))


        if len(inputs) > 1:
            input = T.concatenate(inputs, axis=1)
        else:
            input = inputs[0]

        res = T.dot(input, self._get_shared("W"))

        if self.use_bias:
            res += self._get_shared(name="b")

        if self.activation is not None:
            res = self.activation(res)

        return res





