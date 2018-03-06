from theano import tensor as T

__all__ = [
    'categorical_crossentropy'
]

# ================================================================================== #
# Categorical Cross Entropy Loss

def categorical_crossentropy(coding_dist, true_dist, mask=None):
    """Categorical Cross-entropy

    Calculate cross-entropy given one-hot distribution as indices

    Parameters
    ----------
    coding_dist: 2-D fmatrix
        Each slice along 1-axis represent a distribution
    true_dist: 1-D ivector
        Each element represent the '1' in the one-hot representation
    mask: 1-D fvector
        The mask of true_dist. The default is a vector filled with 1.0.

    Returns
    -------
    ce: vector
        A vector with the same shape as true_dist.
    """

    assert coding_dist.ndim == 2

    n_samples = coding_dist.shape[0]
    n_symbols = coding_dist.shape[1]

    assert true_dist.ndim == 1

    if mask is not None:
        assert mask.ndim == 1
    else:
        mask = T.alloc(1., n_symbols)

    true_dist_flat = T.arange(n_samples) * n_symbols + true_dist

    ce = - T.log(coding_dist.flatten()[true_dist_flat])
    ce = ce * mask

    return ce