### TODO List

#### Short term

- [x] ~~Set disconnected grad to zero grad.~~

~~This is very useful when we only want to optimize part of the graph~~

Already set to "ignore".

- [x] ~~Create a template class for optimizers.~~

All the optimizer has several steps:

    1. get gradients

    2. gradient clip & normalized

    3. accumulated gradients in f_forward.

    4. update parameters & zero gradients.

- [ ] Add doc for the purpose of functional, nn and layers

**functional**: the input of functions is tensor and shared variables,
never create new shared variables inner a function.

**nn**: For modules. Modules consist of functions and shared variables.

**layers**: Consist of one or more modules, do not need to create a
class instance. All of layers is just a python function.

#### Long term

- [ ] tf-like variable scope