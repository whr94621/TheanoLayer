from theano import tensor as T

__all__ = [
    'sparse_softmax_cross_entropy_with_logits',
    'sigmoid_cross_entropy'
]

def sparse_softmax_cross_entropy_with_logits(labels, logits, weights=None):

    assert labels.ndim + 1 == logits.ndim

    cost_shape = labels.shape

    # 1. Reshape logits and labels to rank 2

    if logits.ndim > 2:

        num_classes = logits.shape[-1]

        logits = T.reshape(logits, [-1, num_classes])
        labels = labels.flatten()

    cost = T.nnet.categorical_crossentropy(coding_dist=T.nnet.softmax(logits),
                                           true_dist=labels)

    cost = cost.reshape(cost_shape)

    if weights is not None:

        cost = cost * weights

    return cost

def sigmoid_cross_entropy(labels, probs):

    # TODO:
    loss = labels * (- T.log(probs)) + (1.0 - labels) * (- T.log(1.0 - probs))

    return loss