import numpy as np
from theano import tensor as T
from theanolayer.tensor import array_ops
from theanolayer.initializers import OrthogonalInitializer
from theanolayer.utils import scope, WARNING
from .base import Module

__all__ = [
    'GRUCell'
]

class RNNCellBase(Module):

    def __init__(self,
                 parameters,
                 prefix=""):

        super(RNNCellBase, self).__init__(parameters, prefix)

        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def precal_x(self, *args, **kwargs):
        """Pre-computed the affine transition of input sequence.
        This will gain some performance improvement.
        """
        raise NotImplementedError

class GRUCell(RNNCellBase):

    def __init__(self,
                 parameters,
                 input_size,
                 num_units,
                 activation=T.tanh,
                 transition_depth=1,
                 layer_normalization=False,
                 prefix="gru"):

        super(GRUCell, self).__init__(prefix=prefix, parameters=parameters)

        self.input_size = input_size
        self.num_units = num_units
        self.activation = activation

        self.transition_depth = transition_depth
        self.layer_normalization = layer_normalization

        self._register_params(name="W",
                              value_or_shape=(self.input_size, self.num_units * 3))

        self._register_params(name="b",
                              value_or_shape=(self.num_units * 3, ),
                              )

        self._register_params(name="U",
                              value_or_shape=(self.num_units, self.num_units * 3),
                              initializer=OrthogonalInitializer())

        if self.layer_normalization is True:
            # parameters for ln
            self._register_params(name="r_lnb",
                                  value_or_shape=0.0 * np.ones((self.num_units, )).astype(np.float32)
                                  )

            self._register_params(name="r_lns",
                                  value_or_shape=1.0 * np.ones((self.num_units, )).astype(np.float32)
                                  )

            self._register_params(name="u_lnb",
                                  value_or_shape=0.0 * np.ones((self.num_units, )).astype(np.float32)
                                  )

            self._register_params(name="u_lns",
                                  value_or_shape=1.0 * np.ones((self.num_units, )).astype(np.float32)
                                  )

            self._register_params(name="h_lnb",
                                  value_or_shape=0.0 * np.ones((self.num_units, )).astype(np.float32)
                                  )

            self._register_params(name="h_lns",
                                  value_or_shape=1.0 * np.ones((self.num_units, )).astype(np.float32)
                                  )

        if self.transition_depth > 1:

            for ii in range(1, self.transition_depth):
                self._register_params(name=scope("trans%d" % ii, "b"),
                                      value_or_shape=(self.num_units * 3, ))

                self._register_params(name=scope("trans%d" % ii, "U"),
                                      value_or_shape=(self.num_units, self.num_units * 3),
                                      initializer=OrthogonalInitializer())

            if self.layer_normalization is True:
            # TODO:
            # Add layer norm parameters for transition layer.
                WARNING("LN for transition cells has not support yet.")

    def _ln(self, x, lnb, lns):

        _eps = np.float32(1e-5)

        out = (x - T.mean(x, axis=-1, keepdims=True)) / T.sqrt(T.var(x, axis=-1, keepdims=True) + _eps)
        out = lns * out + lnb

        return out


    def _compute_cell(self, x, h, W, b, U, precal_x, **kwargs):

        # Default not use ln
        if "use_ln" not in kwargs:
            kwargs["use_ln"] = False

        if precal_x is True:
            preact_x = x
        else:
            preact_x = T.dot(x, W) + b

        preact_h = T.dot(h, U)

        r_x, u_x, n_x = array_ops.split(preact_x, split_size=3, axis=-1)
        r_h, u_h, n_h = array_ops.split(preact_h, split_size=3, axis=-1)

        r = r_x + r_h
        if kwargs['use_ln']:
            r = self._ln(r, kwargs['r_lnb'], kwargs['r_lns'])
        r = T.nnet.sigmoid(r)

        u = u_x + u_h
        if kwargs['use_ln']:
            u = self._ln(u, kwargs['u_lnb'], kwargs['u_lns'])
        u = T.nnet.sigmoid(u)

        preact_cand = n_h * r + n_x

        if kwargs['use_ln']:
            preact_cand = self._ln(preact_cand, kwargs['h_lnb'], kwargs['h_lns'])

        h_next = self.activation(preact_cand)

        h_next = (1.0 - u) * h_next + u * h

        return h_next

    def forward(self, x, h, precal_x=False, *args, **kwargs):

        h_next = self._compute_cell(x, h,
                                    self._get_shared("W"), self._get_shared("b"), self._get_shared("U"),
                                    precal_x=precal_x,
                                    use_ln=self.layer_normalization,
                                    r_lnb=self._get_shared(name="r_lnb"), r_lns=self._get_shared(name="r_lns"),
                                    u_lnb=self._get_shared(name="u_lnb"), u_lns=self._get_shared(name="u_lns"),
                                    h_lnb=self._get_shared(name="h_lnb"), h_lns=self._get_shared(name="h_lns")
                                    )

        for ii in range(1, self.transition_depth):

            h_next = self._compute_cell(T.constant(np.float32(0.0), dtype='float32'), h_next, None,
                                        self._get_shared(scope("trans%d" % ii, "b")),
                                        self._get_shared(scope("trans%d" % ii, "U")),
                                        precal_x=True)

        return h_next

    def precal_x(self, x):

        """
        :param x: Input sequence, with shape [max_len, batch_size, input_size] or [batch_size, input_size]

        """

        return T.dot(x, self._get_shared("W")) + self._get_shared("b")

