from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

_DEFAULT_TRNG = RandomStreams(1234)
MINOR = 1e-7

__all__ = [
    'softmax',
    'dropout',
    'lookup',
    'relu',
    'sigmoid',
    'leaky_relu'
]

# Activation functions

def relu(x, alpha=0):
    return T.nnet.relu(x, alpha=alpha)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def leaky_relu(x, alpha=0.01):
    """
    x = x if x > 0 else alpha * x, default x = 0.01

    let leaky_relu(x) = k(x + b|x|)
    x > 0: 1 = k(1 + b)
    x < 0: alpha = k(1 - b)
        => k =
    """
    if alpha != -1.0:
        k = (1.0 + alpha) / 2
        b = (1.0 - alpha) / (1.0 + alpha)

        return k * (x + b * abs(x))
    else:
        return x

def selu(x):
    return T.nnet.selu(x)

# ================================================================================== #
# Softmax

def softmax(input, input_mask=None, axis=-1):

    input_max = T.max(input, axis=axis, keepdims=True)
    input = T.exp(input - input_max)

    if input_mask:
        input = input * input_mask

    outp = input / (T.sum(input, axis=axis, keepdims=True) + MINOR)

    return outp

# ================================================================================== #
# Dropout Layer

def dropout(input,
            keep_prob,
            use_noise,
            trng=_DEFAULT_TRNG,
            ):
    """
    :type trng: MRG_RandomStreams
    """
    if keep_prob == 1.0:
        return input

    noise_shape = input.shape

    random_tensor = keep_prob
    random_tensor += trng.uniform(size=noise_shape, dtype='float32')

    binary_tensor =  T.floor(random_tensor)

    ret = T.switch(use_noise,
                   input / keep_prob * binary_tensor,
                   input
                   )

    return ret

# ================================================================================== #
# Lookup Table
def lookup(x, W):
    x_shp = x.shape
    emb = W[x.flatten()]

    emb_shp = T.concatenate([x_shp, [-1]], axis=0)
    emb = emb.reshape(emb_shp, ndim=(x.ndim + 1))

    return emb
