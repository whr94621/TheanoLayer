# TheanoLayer: A light theano library

TheanoLayer is a light library written with theano. It is conveninet for
fast building and training DL models. Since [MILA has stop developing theano](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ) and
tensorflow is the most dominant static graph library， I still find that theano is more speed and memory efficiency for
some models, especially seq2seq model in NLP which heavily use for-loop structure.

## Features

- Convenient shared variables creation, sharing and serilization.

- Sparse gradient update for lookup table.

- Memory efficient optimizer

## Concept

- **functional**: Build compatition graph given input tensors and shared
variables. Like functions in python.

- **Module**: A combination of functionals and shared variables. Modules
are all python classes which inherit a base class Module.

- **layer**: A combination of different modules. It is a high-level
interface for some frequent-used layers when building networks, such
as *dense* layer or *bidirectional gru* layer.

## Code Stuctures

- **utils**: utility functions

- **nnet**: funtionals (build computation) and modules

- **layers**: neural layers

- **parameters.py**: A global shared variables management mechanism.

- **variable.py**: A wrapper of theano shared variable for gradients
accumulation and sparse update.

- **optimizers.py**: Gradient optimizer algorithms.

## Acknowledgment

This library borrows some codes from [Lasagne](https://github.com/Lasagne/Lasagne) and [Tensorflow](https://github.com/tensorflow/tensorflow).

## Contact

Haoran Wei(whr94621@163.com, whr94621@gmail.com)



