# author: Hao-ran Wei
# e-mail: whr94621@gmail.com

import numpy as np
import math

__all__ = [
    'ConstantInitializer',
    'NormalInitializer',
    'OrthogonalInitializer',
    'UniformInitializer',
    'XavierInitializer',
    'ZeroInitilizer',
]

def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    tanh         :math:`5 / 3`
    relu         :math:`\sqrt{2}`
    leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    ============ ==========================================

    Args:
        nonlinearity: the nonlinear function (`nn.functional` name)
        param: optional parameter for the nonlinear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

class Initializer(object):
    def __init__(self, dtype='float32'):
        self.dtype = dtype
        # raise NotImplementedError

    def __call__(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, (list, tuple))

        return self._set_value(shape)

    def _set_value(self, shape):
        raise NotImplementedError

class UniformInitializer(Initializer):
    def __init__(self, limit=0.1, dtype='float32'):
        super(UniformInitializer, self).__init__(dtype)

        self.limit = limit

    def _set_value(self, shape):

        return np.random.uniform(low=-self.limit, high=self.limit, size=shape).astype(self.dtype)


class NormalInitializer(Initializer):
    def __init__(self, scale=0.01, dtype='float32'):
        super(NormalInitializer, self).__init__(dtype)
        self.scale = scale

    def _set_value(self, shape):
        return np.random.randn(*shape).astype(self.dtype) * self.scale

class ConstantInitializer(Initializer):
    def __init__(self, value, dtype='float32'):
        super(ConstantInitializer, self).__init__(dtype)
        self.value = value

    def _set_value(self, shape):
        return np.ones(shape=shape, dtype=self.dtype) * self.value


def ZeroInitilizer(dtype='float32'):
    return ConstantInitializer(value=0, dtype=dtype)


class OrthogonalInitializer(Initializer):
    """
    dl4mt-style orthogonal initializer.
    Only for 2-D array
    """
    def __init__(self, scale=1.0, dtype='float32'):
        super(OrthogonalInitializer, self).__init__(dtype)
        self.scale = scale

    def _set_value(self, shape):
        if len(shape) != 2:
            raise ValueError("The variable to initialize must be "
                             "two-dimensional."
                             )

        if shape[0] % shape[1] == 0:
            axis = 0
            n = shape[0] // shape[1]
            sub_shp = (shape[1], shape[1])
        elif shape[1] % shape[0] == 0:
            axis = 1
            n = shape[1] // shape[0]
            sub_shp = (shape[0], shape[0])
        else:
            raise ValueError("""
                The shape must meet:
                    shape[0] % shape[1] == 0
                or
                    shape[1] % shape[0] == 0
            """)

        sub_tensors = [np.random.randn(*sub_shp) for _ in range(n)]
        sub_tensors = [np.linalg.svd(w)[0] for w in sub_tensors]

        return np.concatenate(sub_tensors, axis=axis).astype(self.dtype)

class XavierInitializer(Initializer):

    def __init__(self, dtype='float32'):

        super(XavierInitializer, self).__init__(dtype)

    def _set_value(self, shape):

        n_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        n_out = float(shape[-1])

        limit = math.sqrt(6.0 / (n_in + n_out))

        return np.random.uniform(low=-limit, high=limit, size=shape).astype(self.dtype)

def _calculate_fin_in_and_fan_out(shape, mode="fan_in"):

    if len(shape) < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if len(shape) == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]

        receptive_field_size = 1

        if len(shape) > 2:
            receptive_field_size = np.prod(np.array(list(shape)[2:]))
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in if mode == "fan_in" else fan_out

class KaimingUniformInitializer(Initializer):

    def __init__(self, mode="fan_in", nolinearity="leaky_relu", dtype="float32"):

        super(KaimingUniformInitializer, self).__init__(dtype=dtype)

        self.mode = mode
        self.nolinearity = nolinearity

    def _set_value(self, shape):

        fan = _calculate_fin_in_and_fan_out(shape=shape, mode=self.mode)

        gain = calculate_gain(nonlinearity=self.nolinearity, param=0)

        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        return np.random.uniform(-bound, bound, size=shape).astype('float32')


_DEFAULT_RAND_INIT = XavierInitializer()
_DEFAULT_NORMAL_INIT = NormalInitializer()
_DEFAULT_ZERO_INIT = ZeroInitilizer()