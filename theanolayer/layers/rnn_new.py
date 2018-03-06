import theano
from theano import tensor as T
from theanolayer.nnet.rnn_cell import GRUCell
from theanolayer.utils import scope

def unidirectional_gru_layer(parameters,
                             input,
                             input_size,
                             num_units,
                             input_mask=None,
                             init_state=None,
                             activation=T.tanh,
                             transition_depth=1,
                             one_step=False,
                             layer_norm=False,
                             prefix='gru_layer'):

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


    gru_cell = GRUCell(parameters=parameters,
                       input_size=input_size,
                       num_units=num_units,
                       activation=activation,
                       transition_depth=transition_depth,
                       prefix=prefix,
                       layer_normalization=layer_norm
                       )

    precal_x = gru_cell.precal_x(input)

    all_shared =  gru_cell.pack_params()[2:]

    def _step(mask, precal_x,
              prev_h,
              *args):

        args = list(args)
        all_params, args = gru_cell.unpack_params(args, num_args=gru_cell.num_params - 2)
        all_params = [None, None] + all_params

        next_h = gru_cell(x=precal_x, h=prev_h, precal_x=True, params=all_params)

        next_h = mask[:,None] * next_h + (1.0 - mask)[:,None] * prev_h

        return next_h

    res, _ = theano.scan(fn=_step,
                         sequences=[input_mask, precal_x],
                         outputs_info=[init_state],
                         non_sequences=all_shared,
                         n_steps=n_timesteps,
                         strict=False,
                         return_list=True
                         )

    return res


def bidirectional_gru_layer(parameters,
                            input,
                            input_size,
                            num_units,
                            input_mask=None,
                            init_state=None,
                            transition_depth=1,
                            layer_norm=False,
                            activation=T.tanh,
                            prefix='gru_layer'):


    res_fw = unidirectional_gru_layer(parameters=parameters,
                                      input=input,
                                      input_size=input_size,
                                      num_units=num_units,
                                      input_mask=input_mask,
                                      init_state=init_state,
                                      activation=activation,
                                      transition_depth=transition_depth,
                                      one_step=False,
                                      layer_norm=layer_norm,
                                      prefix=scope(prefix, "fwd"))

    res_bw = unidirectional_gru_layer(parameters=parameters,
                                      input=input[::-1],
                                      input_size=input_size,
                                      num_units=num_units,
                                      input_mask=input_mask[::-1],
                                      init_state=init_state,
                                      activation=activation,
                                      transition_depth=transition_depth,
                                      one_step=False,
                                      layer_norm=layer_norm,
                                      prefix=scope(prefix, "bwd"))


    return [res_fw[0], res_bw[0][::-1]]

