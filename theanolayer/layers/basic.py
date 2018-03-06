from theano import tensor as T
import numpy as np

from theanolayer.initializers import UniformInitializer
from theanolayer.parameters import Parameters
from theanolayer.nnet.linear import Dense
from theanolayer.nnet.sparse import Embedding
from theanolayer.nnet.norm import LayerNorm

__all__ = [
    'dense',
    'layer_norm'
]

def dense(parameters, inputs,
          input_size,
          num_units,
          activation=None,
          use_bias=True,
          prefix='dense',
          kernel_initializer=None,
          bias_initializer=None
          ):
    """
    :type parameters: Parameters
    :param parameters: Parameters instance.
    """

    _dense = Dense(input_size=input_size,
                   num_units=num_units,
                   parameters=parameters,
                   use_bias=use_bias, activation=activation, kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer, prefix=prefix)

    res = _dense(inputs)

    return res

# ================================================================================== #
# Layer Normalization

def layer_norm(parameters,
               input,
               input_size,
               prefix="layer_norm"
               ):
    """ Layer Normalization

    Apply layer normalization to the input before applying the non-linearity. See more
    details in https://arxiv.org/abs/1607.06450.

    Parameters
    ----------
    parameters
    input: 2-D or 3-D tensor
        Input. Here we only support 2-D and 3-D tensor.
    input_size: int
        Input size.
    prefix: str
        Scope prefix of the layer normalization parameters.

    Returns
    -------
    output: tensor
        Layer normalized input.
    """

    _layer_norm = LayerNorm(parameters=parameters, input_size=input_size, prefix=prefix)

    return _layer_norm(input)


# ================================================================================== #
# Look-up Table Layer

def embedding_layer(parameters,
                    input,
                    num_symbols,
                    num_units,
                    prefix="embedding",
                    sparse_update=False,
                    initializer=UniformInitializer()
                    ):

    _layer = Embedding(parameters=parameters,
                       num_symbols=num_symbols,
                       num_units=num_units,
                       prefix=prefix,
                       sparse_update=sparse_update,
                       initializer=initializer
                       )

    res = _layer(input)

    return res
