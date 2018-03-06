# ================================================================================== #
# Full Connected Layer
# A wrapper of theano shared variable

from theano import tensor as T

"""
Variables:
    :  of the variable
    initializer: give a  return a numpy ndarray for initial value.
    shared:
    shared_grad:
    name:
    indices: slicing by tensor
    subset: the result of the slice. indices and subset are vital for implementing sparse update.

Properties:
    value:
    grad:
    sparse_updated: boolean. whether do sparse update.
Methods:
"""

import numpy as np
from theano import shared
from .initializers import _DEFAULT_RAND_INIT, _DEFAULT_ZERO_INIT

__all__ = ['Variable']


class Variable(object):
    def __init__(self,
                 name,
                 value_or_shape,
                 initializer=None,
                 frozen=False
                 ):


        if isinstance(value_or_shape, (list, tuple)):

            if initializer is None:
                if isinstance(value_or_shape, int) or len(value_or_shape) == 1:
                    initializer = _DEFAULT_ZERO_INIT
                else:
                    initializer = _DEFAULT_RAND_INIT

            data = initializer(value_or_shape)
        else:
            data = value_or_shape

        grad = np.zeros_like(data)

        self.shared = shared(value=data, name=name)
        if frozen is False:
            self.grad = shared(value=grad, name='%s_%s' % (name, 'grad'))
        else:
            self.grad = None

        self.name = name

        self.frozen = frozen # whether update or not

        self.update_index = None
        self.update_subset = None

    @property
    def data(self):
        return self.shared.get_value()

    @property
    def shape(self):
        return self.shared.get_value(borrow=True).shape

    @property
    def dtype(self):
        return self.shared.get_value(borrow=True).dtype

    @property
    def broadcastable(self):
        return self.shared.broadcastable

    @property
    def trainable(self):
        """
        What is the trainable of a variable.
        If a variable is frozen, the trainable is false.
        If a variable should be updated partially, the trainable is its updated subset.
        Else, the trainable should be its shared
        """
        if self.frozen is True:
            return None
        elif self.update_index is not None and self.update_subset is not None:
            return self.update_subset
        else:
            return self.shared

    def get_value(self, borrow=False):
        return self.shared.get_value(borrow=borrow)

    def read(self, new_data):
        self.shared.set_value(new_data)

    def zero_grad(self):
        grad = np.zeros_like(self.data)
        self.grad.set_value(grad)

    def partial_update(self, index, subset):
        self.update_index = index
        self.update_subset = subset

    def accumulate_grad(self, accu_grad):
        if self.update_index is not None:
            return (self.grad, T.inc_subtensor(self.grad[self.update_index], accu_grad))
        else:
            return (self.grad, self.grad + accu_grad)
