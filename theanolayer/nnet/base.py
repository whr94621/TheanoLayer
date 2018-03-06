from theanolayer.utils.common_utils import scope
from theanolayer.utils import nest
from theanolayer.parameters import Parameters
from collections import OrderedDict

class Module(object):

    def __init__(self, parameters, prefix=""):

        self._parameters = parameters # type: Parameters
        self._prefix = prefix

    @property
    def prefix(self):
        return self.prefix

    @property
    def param_names(self):
        return list(self._params.keys())

    @property
    def num_params(self):

        return len(self.param_names)

    def pack_params(self):

        return [self.__getattribute__(name) for name in self.param_names]

    def op_scope(self, name):

        if name == "" or name is None:
            return self._prefix

        return scope(self._prefix, name=name)

    def unpack_params(self, args, num_args=None):

        if num_args is None:
            num_args = self.num_params

        if num_args > self.num_params:
            raise ValueError("Module {0} only have {1} parameters but expect {2}".format(self.prefix, self.num_params, num_args))

        module_params = args[:num_args]

        return module_params, args[num_args:]


    def _register_params(self, name, value_or_shape, initializer=None):
        """
        When to use register params

        Register parameters in a Module makes it easy to import and export parameters, which is especially
        useful to use modules in a theano scan as theano propose to pass shared variable within a scan loop
        as 'non-sequences'.

        """
        if "_params" not in self.__dict__:
            self._params = OrderedDict()

        shared = self._parameters.get_shared(name=self.op_scope(name),
                                             value_or_shape=value_or_shape,
                                             initializer=initializer)

        # When use Op in the scan loop,
        # It is more efficiency to pass parameter from non-sequence
        # We could put those non-sequence parameters in this dict by name
        self._params[name] = None

        self.__setattr__(name, shared)

    def _update_params(self, params):
        """
        Update parameters by order. If no update at that place, just use a None.

        """
        assert len(params) == self.num_params

        for i in range(len(params)):
            self._params[self.param_names[i]] = params[i]

    def _get_shared(self, name):

        if name not in self._params:
            return None

        if self._params[name] is not None:
            return self._params[name]
        else:
            return self.__getattribute__(name)


    def __call__(self, *args, **kwargs):

        if "params" in kwargs:
            params = nest.flatten(kwargs['params'])
            self._update_params(params)

        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):

        raise NotImplementedError