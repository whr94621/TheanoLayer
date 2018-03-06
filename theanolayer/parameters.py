import re
import theano
from theano import tensor as T
from collections import OrderedDict
from .variable import Variable
from .utils import ERROR

__all__ = ['Parameters']

"""
Parameters
----------

Parameters is a container to construct, update, save and load theano SharedVariables.

TODO:
1. Reuse Mechanism

get_shared should have a option 'reuse' to decide whether to use or cover the exist
shared variables.

2. Freeze Mechanism

Variables could be chosen whether be frozen or updated.

3. Partial Update (medium priority)

Update the whole shared variable or just a subset of it.
This mechanism will greatly improve the efficiency of large lookup table.
"""

class Exclude(object):
    def __init__(self, exclude=None):
        if exclude is None:
            self.exclude = None
        else:
            if not isinstance(exclude, (list, tuple)):
                exclude = [exclude]
            self.exclude = []
            for p in exclude:
                self.exclude.append(re.compile(r'{0}'.format(p)))

    def __call__(self, string):
        if self.exclude is None:
            return False
        for p in self.exclude:
            if p.search(string) is not None:
                return True
        return False

class Parameters(object):
    def __init__(self, load_new=False):

        self.kv_store = OrderedDict()
        self.load_new = load_new

    def __getitem__(self, key):
        if key not in self.kv_store:
            raise KeyError
        else:
            return self.kv_store[key]

    def __setitem__(self, key, value):
        assert isinstance(value, Variable)

        if key in self.kv_store:
            raise KeyError
        else:
            self.kv_store[key] = value

    def __contains__(self, key):
        return key in self.kv_store

    def get_shared(self,
                   name,
                   value_or_shape=None,
                   initializer=None,
                   frozen=False
                  ):
        """
        Create or reuse a shared variable
        """

        if name in self.kv_store:
            return self.kv_store[name].shared
        else:
            assert value_or_shape is not None
            variable = Variable(name=name,
                                value_or_shape=value_or_shape,
                                initializer=initializer,
                                frozen=frozen
                                )
            self.add(variable)

            return variable.shared

    @property
    def all_varaibles(self):
        """
        Return a list of all variables
        """
        return [var for var in self.kv_store.values()]

    @property
    def all_variables_name(self):
        """
        Return all the keys's of variables in a list
        """
        return [name for name in self.kv_store.keys()]

    def reload_value(self, params, exclude):

        exc = Exclude(exclude=exclude)
        for name, param in params.items():
            if exc(name) is False and name in self.kv_store:
                self.kv_store[name].shared.set_value(param)

    def export(self):
        """
        Export data of all variables
        """
        params = OrderedDict()

        for name, param in self.kv_store.items():
            params[name] = param.data

        return params

    def load(self, params, exclude=None):

        exc = Exclude(exclude=exclude)
        for name, param in params.items():
            if exc(name) is False:
                self.kv_store[name] = Variable(name=name, value_or_shape=param)


    def add(self, variable):

        assert isinstance(variable, Variable), 'Must be a Varaible!'

        self.kv_store[variable.name] = variable

    def grad(self, loss, exclude_grad=None):
        """
        Get the gradients of all the trainable variables(var.froze is False)
        """

        if exclude_grad is not None:
            exclude_grad = Exclude(exclude=exclude_grad)

        def _not_grad(var):

            if var.frozen is True:
                return True

            if exclude_grad is not None and exclude_grad(var.name) is True:
                return True

            return False

        all_trainable_vars = [var for var in self.all_varaibles if _not_grad(var) is False]

        gradients = []

        try:
            gradients = T.grad(cost=loss, wrt=[var.trainable for var in all_trainable_vars],
                               disconnected_inputs="ignore")
        except:
            for var in all_trainable_vars:
                try:
                    _ = T.grad(cost=loss, wrt=[var.trainable])
                except:
                    ERROR('{0} has wrong gradient when computing {1}!'.format(var.name, loss))

            exit(5)

        return all_trainable_vars, gradients

    def freeze(self, name=None, startswith=None, contain=None):
        """Freeze a variable

        name: by name
        startswith: by prefix
        contain: by pattern
        """

        if contain is not None:
            p_contain = re.compile(contain)
        else:
            p_contain = None

        for key in self.all_variables_name:
            if name is not None and key == name:
                self.kv_store[key].frozen = True
                continue

            if startswith is not None and key.startswith(startswith):
                self.kv_store[key].frozen = True
                continue

            if p_contain is not None and p_contain.search(key):
                self.kv_store[key].frozen = True



