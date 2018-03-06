import theano
from theano import tensor as T
from theanolayer.tensor import array_ops
from theanolayer.utils.common_utils import scope
from theanolayer.initializers import *
from .basic import layer_norm

__all__ = [
    'gru_layer'
]

# ================================================================================== #
# GRU Layer

def _gru_cell_params(parameters, input_size, num_units, prefix='gru_cell'):

    if input_size != 0:
        W = parameters.get_shared(name=scope(prefix, 'W'),
                                  value_or_shape=(input_size, num_units * 2))
    else:
        W = None


    U = parameters.get_shared(name=scope(prefix, 'U'),
                              value_or_shape=(num_units, num_units * 2),
                              initializer=OrthogonalInitializer()
                              )

    b = parameters.get_shared(name=scope(prefix, 'b'),
                              value_or_shape=(num_units * 2,)
                              )

    if input_size != 0:
        Wx = parameters.get_shared(name=scope(prefix, 'Wx'),
                                  value_or_shape=(input_size, num_units)
                                  )
    else:
        Wx = None

    Ux = parameters.get_shared(name=scope(prefix, 'Ux'),
                               value_or_shape=(num_units, num_units),
                               initializer=OrthogonalInitializer()
                              )

    bx = parameters.get_shared(name=scope(prefix, 'bx'),
                              value_or_shape=(num_units,)
                              )

    return W, U, b, Wx, Ux, bx

def _gru_cell(parameters,
              h_,
              U, Ux,
              activation,
              input=None, inputx=None,
              W=None, b=None, Wx=None, bx=None,
              layer_normalization=False
              ):

    preact = T.dot(h_, U)

    if b is not None:
        preact += b

    if layer_normalization is True:
        preact = layer_norm(parameters=parameters,
                            input=preact,
                            input_size=U.get_value(borrow=True).shape[1],
                            scope=U.name
                            )

    if input is not None:
        if W is not None:
            state_below = T.dot(input, W)


            if layer_normalization is True:
                state_below = layer_norm(parameters=parameters,
                                         input=state_below,
                                         input_size=W.get_value(borrow=True).shape[-1],
                                         scope=W.name
                                         )
        # If W is None
        # We assume that layer norm has been pre-computed
        # outside the GRU cell.
        else:
            state_below = input

        preact += state_below


    preact = T.nnet.sigmoid(preact)
    r, u = array_ops.split(preact, axis=-1, split_size=2)

    preactx = T.dot(h_, Ux)

    if bx is not None:
        preactx += bx

    if layer_normalization is True:
        preactx = layer_norm(parameters=parameters,
                             input=preactx,
                             input_size=Ux.get_value(borrow=True).shape[-1],
                             scope=Ux.name
                             )

    preactx *= r

    if inputx is not None:
        if Wx is not None:
            state_belowx = T.dot(input, Wx)

            if layer_normalization is True:
                state_belowx = layer_norm(parameters=parameters,
                                          input=state_belowx,
                                          input_size=Wx.get_value(borrow=True).shape[-1],
                                          scope=Wx.name
                                          )

        # If W is None
        # We assume that layer norm has been pre-computed
        # outside the GRU cell.
        else:
            state_belowx = inputx

        preactx += state_belowx

    h = activation(preactx)
    h = u * h_ + (1. - u) * h

    return h

def gru_layer(parameters,
              input,
              input_size,
              num_units,
              input_mask=None,
              init_state=None,
              activation=T.tanh,
              layer_normalization=False,
              transition_depth=1,
              one_step=False,
              prefix='gru_layer'
              ):

    if not callable(activation):
        raise ValueError("activation must be callable.")

    n_timesteps = input.shape[0]

    if input.ndim == 3:
        batch_size = input.shape[1]
    else:
        batch_size = input.shape[0]

    if input_mask is None:
        if one_step is False:
            input_mask = T.alloc(1., n_timesteps, batch_size)
        else:
            input_mask = T.alloc(1., batch_size)

    if init_state is None:
        assert one_step is False
        init_state = T.alloc(0., batch_size, num_units)

    # ================================ #
    # GRU Layer Parameters
    params = []

    # 0-th layer
    params_0 = \
        _gru_cell_params(parameters=parameters,
                         input_size=input_size,
                         num_units=num_units,
                         prefix=prefix)

    params.append(params_0)

    # deep transition layer
    for i in range(transition_depth - 1):
        _, U, b, _,Ux, bx = \
            _gru_cell_params(parameters=parameters,
                             input_size=0,
                             num_units=num_units,
                             prefix=scope(prefix, 'trans_layer%d' % (i + 1))
                             )
        params.append([U,b,Ux,bx])

    state_below = T.dot(input, params[0][0]) + params[0][2]
    state_belowx = T.dot(input, params[0][3]) + params[0][5]

    if layer_normalization:
        state_below = layer_norm(parameters=parameters,
                                 input=state_below,
                                 input_size=params[0][2].get_value(borrow=True).shape[-1],
                                 scope=params[0][0].name
                                 )

        state_belowx = layer_norm(parameters=parameters,
                                 input=state_belowx,
                                 input_size=params[0][5].get_value(borrow=True).shape[-1],
                                 scope=params[0][3].name
                                 )

    def _step_slice(m_, x_, xx_, h_,
                    U, Ux,
                    *args
                    ):
        """
        args:
        [U_lnb, U_lns, Ux_lnb, Ux_lns]
        """


        h = _gru_cell(activation=activation,
                      h_=h_, U=U, Ux=Ux,
                      input=x_, inputx=xx_,
                      parameters=parameters,
                      layer_normalization=layer_normalization
                      )

        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        h_prev = h

        for i in range(transition_depth - 1):
            h = _gru_cell(parameters=parameters,
                          h_=h_prev,
                          U=args[i * 4 + 0],
                          b=args[i * 4 + 1],
                          Ux=args[i * 4 + 2],
                          bx=args[i * 4 + 3],
                          activation=activation,
                          layer_normalization=layer_normalization
                          )
            h = m_[:, None] * h + (1. - m_)[:, None] * h_prev

            h_prev = h

        return h

    # prepare scan arguments
    seqs = [input_mask, state_below, state_belowx]
    init_states = [init_state]
    _step = _step_slice

    shared_vars = [params[0][1], params[0][4]]

    for i in range(transition_depth - 1):
        shared_vars += params[i + 1]

    if one_step is False:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=init_states,
                                    non_sequences=shared_vars,
                                    name=scope(prefix, 'scan'),
                                    n_steps=n_timesteps,
                                    strict=False
                                    )
    else:
        rval = _step_slice(input_mask, state_below, state_belowx, init_state,
                           *shared_vars)

    rval = [rval]
    return rval


