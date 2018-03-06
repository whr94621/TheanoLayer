import numpy as np
import theano
from theano import tensor as T

from .variable import Variable
from theanolayer.utils import nest
from collections import OrderedDict
from .utils import nest

__all__ = [
    'apply_regularization',
    'build_optimizer'
]

def l2_norm(param):
    return (param ** 2).sum()

def apply_regularization(parameters, decay_c, method='l2'):

    if method == 'l2':
        method = l2_norm
    else:
        raise ValueError

    decay_c = theano.shared(value=np.float32(decay_c), name='decay_c')
    weight_decay = 0

    for _, var in parameters.get_params():
        weight_decay += method(var.shared)

    weight_decay *= decay_c

    return weight_decay

def apply_gradient_clip(grads, clip_c):
    g2 = 0.
    for g in grads:
        g2 += (g ** 2).sum()
    new_grads = []
    for g in grads:
        new_grads.append(T.switch(g2 > (clip_c ** 2),
                                       g / T.sqrt(g2) * clip_c,
                                       g))
    grads = new_grads

    return grads


class Optimizer(object):
    """
    let Loss = Loss1 + Loss2 + ... + Lossn,

    We have that dLoss/dv = dLoss1/dv + dLoss2/dv + ... + dLossn/dv.

    For memory efficiency, we can compute dLoss1/dv, ... dLossn/dv respectively,
    accumulate those gradients to dLoss/dv and update variable v for one time.

    How does optimizer process several loss?

    For each loss:
        1. compute gradients w.r.t variables.
        2. build f_fwd for those variables to update their gradients.

    For all variables computed in those f_fwd:

        1. build optimizer algorithm to update their value.
        2. clean their gradient.
    """
    def __init__(self,
                 parameters,
                 loss, inputs,
                 clip_gradient=None,
                 aux_outputs=None,
                 exclude=None, **kwargs):

        """
        :param exclude: Patterns on those variables whose are not updated.
        """

        self.parameters = parameters
        self.loss = nest.flatten(loss)

        if len(self.loss) == 1:
            self.inputs = [inputs]
        else:
            self.inputs = inputs

        self.clip_grad = clip_gradient

        if aux_outputs is not None:
            if not nest.is_sequence(aux_outputs):
                raise ValueError("aux_outputs must be nested.")

            if len(self.loss) == 1:
                self.aux_outputs = [aux_outputs]
            else:
                aux_outputs_ = []

                self.aux_outputs = [aux_outputs_[ii] if ii < len(aux_outputs_) else None for ii in range(len(self.loss))]

        else:
            self.aux_outputs = [None for _ in range(len(self.loss))]


        self.lrate = T.scalar(name='lrate', dtype='float32')

        if exclude is None:
            exclude = [None for _ in range(self.n_loss)]

        self.exclude = exclude

        # Initialize optimizer
        self.init_optimizer()

    @property
    def n_loss(self):

        return len(self.loss)

    @property
    def all_trainable_variables(self):
        """
        All the variables used to compute gradients.
        """
        all_trainable_variables_dict = OrderedDict()

        for ii in range(self.n_loss):
            for var in self.all_trainable_vars_queue[ii]: # type: Variable
                if var.name not in all_trainable_variables_dict:
                    all_trainable_variables_dict[var.name] = var

        return list(all_trainable_variables_dict.values())


    def init_optimizer(self):

        self.all_trainable_vars_queue = []  # trainable variables per loss
        self.accu_grads_queue = []


    def compute_gradients(self):

        for ii, ll in enumerate(self.loss):

            trainable_vars, accu_grads = self.parameters.grad(ll, exclude_grad=self.exclude[ii])

            if self.clip_grad is not None:
                accu_grads = apply_gradient_clip(accu_grads, clip_c=self.clip_grad)

            self.accu_grads_queue.append(accu_grads)

            self.all_trainable_vars_queue.append(trainable_vars)

    def build_func_forwards(self):

        f_forwards = []

        for ii in range(self.n_loss):
            updates_grad = []
            for var, accu_grad in zip(self.all_trainable_vars_queue[ii], self.accu_grads_queue[ii]):
                updates_grad.append(var.accumulate_grad(accu_grad))

            f_fwd = theano.function(inputs=self.inputs[ii],
                                    outputs=self.loss[ii] if self.aux_outputs[ii] is None else self.loss[ii] + self.aux_outputs[ii],
                                    updates=updates_grad,
                                    name="f_fwd_%d" % ii)
            f_forwards.append(f_fwd)

        return f_forwards

    def build_backward_updates(self):

        raise NotImplementedError

    def zero_grads(self):

        return [(var.grad, var.grad * 0.0) for var in self.all_trainable_variables]

    def __call__(self):

        self.compute_gradients()

        f_forwards =  self.build_func_forwards()

        updates = self.build_backward_updates()

        updates = updates + self.zero_grads()

        f_backward = theano.function(inputs=[self.lrate],
                                     outputs=None,
                                     updates=updates,
                                     name="f_backward", on_unused_input="ignore")

        return f_forwards + [f_backward]


class AdamOptimizer(Optimizer):

    def __init__(self, parameters, loss, inputs,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 clip_gradient=None,
                 aux_outputs=None,
                 **kwargs):

        super(AdamOptimizer, self).__init__(parameters, loss, inputs, clip_gradient, aux_outputs=aux_outputs, **kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def build_backward_updates(self):
        updates = []

        t_prev = theano.shared(value=np.float32(0.))
        one = T.constant(1.)
        t = t_prev + 1.0

        a_t = self.lrate * T.sqrt(one - self.beta2 ** t) / (one - self.beta1 ** t)

        for var in self.all_trainable_variables:
            # ================================================================================== #
            # Optimization Procedure
            value = var.get_value(borrow=True)

            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=var.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=var.broadcastable)

            m_t = self.beta1 * m_prev + (one - self.beta1) * var.grad
            v_t = self.beta2 * v_prev + (one - self.beta2) * var.grad ** 2

            step = a_t * m_t / (T.sqrt(v_t) + self.epsilon)

            updates.append((m_prev, m_t))
            updates.append((v_prev, v_t))
            updates.append((var.shared, var.shared - step))

        updates.append((t_prev, t))

        return updates

class AdadeltaOptimizer(Optimizer):

    def __init__(self, parameters, loss, inputs,
                 rho=0.95,
                 epsilon=1e-6,
                 clip_gradient=None,
                 aux_outputs=None, **kwargs):

        super(AdadeltaOptimizer, self).__init__(parameters, loss, inputs, clip_gradient, aux_outputs=aux_outputs, **kwargs)

        self.rho = rho
        self.epsilon = epsilon

    def build_backward_updates(self):
        updates = []

        for var in self.all_trainable_variables:
            # ================================================================================== #
            # Optimization Procedure

            accu = theano.shared(np.zeros(var.shape, dtype=var.dtype), broadcastable=var.broadcastable)

            delta_accu = theano.shared(np.zeros(var.shape, dtype=var.dtype), broadcastable=var.broadcastable)

            accu_new = self.rho * accu + (1. - self.rho) * var.grad ** 2

            updates.append((accu, accu_new))

            update = var.grad * T.sqrt(delta_accu + self.epsilon) / T.sqrt(accu_new + self.epsilon)

            updates.append((var.shared, var.shared - 1.0 * update))

            delta_accu_new = self.rho * delta_accu + (1. - self.rho) * update ** 2
            updates.append((delta_accu, delta_accu_new))

        return updates

class RMSPropOptimizer(Optimizer):

    def __init__(self, parameters, loss, inputs,
                 rho=0.9,
                 epsilon=1e-6,
                 clip_gradient=None,
                 aux_outputs=None, **kwargs):

        super(RMSPropOptimizer, self).__init__(parameters, loss, inputs, clip_gradient, aux_outputs=aux_outputs, **kwargs)

        self.rho = rho
        self.epsilon = epsilon


    def build_backward_updates(self):

        updates = []

        for var in self.all_trainable_variables:
            # ================================================================================== #
            # Optimization Procedure

            accu = theano.shared(value=np.zeros(var.shape, dtype=var.dtype), broadcastable=var.broadcastable)

            accu_new = self.rho * accu + (1.0 - self.rho) * var.grad ** 2

            updates.append((accu, accu_new))
            updates.append((var.shared, var.shared - (self.lrate * var.grad / T.sqrt(accu_new + self.epsilon))))

        return updates

class SGDOptimizer(Optimizer):

    def __init__(self, parameters, loss, inputs,
                 clip_gradient=None, aux_outputs=None, **kwargs):

        super(SGDOptimizer, self).__init__(parameters, loss, inputs,
                                           clip_gradient=clip_gradient,
                                           aux_outputs=aux_outputs, **kwargs)


    def build_backward_updates(self):
        updates = []

        for var in self.all_trainable_variables:

            # ================================================================================== #
            # Optimization Procedure

            updates.append((var.shared, var.shared - self.lrate * var.grad))

        return updates

OPTIMIZERS = {
    "adam": AdamOptimizer,
    "sgd": SGDOptimizer,
    "adadelta": AdadeltaOptimizer,
    "rmsprop": RMSPropOptimizer
}

def build_optimizer(optimizer,
                    parameters,
                    loss,
                    inputs,
                    clip_gradient=None,
                    aux_outputs=None,
                    **kwargs):

    """
    :type optimizer: str
    """

    if not isinstance(optimizer, str) or optimizer not in OPTIMIZERS:
        raise ValueError("Unknown optimizer name {0}".format(optimizer))


    return OPTIMIZERS[optimizer](parameters=parameters,
                               loss=loss,
                               inputs=inputs,
                               clip_gradient=clip_gradient,
                               aux_outputs=aux_outputs,
                               **kwargs
                               )()



