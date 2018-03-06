from theanolayer.nnet.functional import lookup
from .base import Module

__all__ = [
    'Embedding'
]

class Embedding(Module):

    def __init__(self, parameters,
                 num_symbols,
                 num_units,
                 sparse_update,
                 initializer=None,
                 prefix="embedding"):

        super(Embedding, self).__init__(parameters, prefix)

        self.num_symbols = num_symbols
        self.num_units = num_units
        self.sparse_update = sparse_update

        self.emb = self._register_params(name="weight",
                                         value_or_shape=(self.num_symbols, self.num_units),
                                         initializer=initializer)


    def __call__(self, indices, *args, **kwargs):

        res = lookup(indices, self._get_shared(name="weight"))

        if self.sparse_update is True:

            self._parameters[self.op_scope(name="weight")].parrtial_update(indices, res)

        return res


