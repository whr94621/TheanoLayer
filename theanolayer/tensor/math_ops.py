from functools import reduce

__all__ = [
    'add_n'
]

def add_n(inputs):

    if len(inputs) == 1:
        return inputs[0]

    else:
        return reduce(lambda x, y: x + y, inputs)