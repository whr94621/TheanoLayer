#!/usr/bin/env python
from distutils.core import setup

setup(name='TheanoLayer',
      version='0.1.dev0',
      description='A light theano wrapper.',
      author='Hao-Ran Wei',
      author_email='whr94621@gmail.com',
      packages=[
          'theanolayer',
          'theanolayer.layers',
          'theanolayer.tensor',
          'theanolayer.utils',
          'theanolayer.nnet',
          'theanolayer.nnet.functional'
      ]
)