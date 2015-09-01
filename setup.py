#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
setup(
  name = 'nnet-ts',
  packages = ['nnet-ts'], # this must be the same as the name above
  version = '0.2',
  description = 'Neural network architecture for time series forecasting.',
  author = 'José Jiménez',
  author_email = 'jose@jimenezluna.com',
  url = 'https://github.com/hawk31/nnet-ts', # use the URL to the github repo
  download_url = 'https://github.com/hawk31/nnet-ts/archive/master.zip', # I'll explain this in a second
  keywords = ['time series', 'neural network', 'machine learning'], # arbitrary keywords
  classifiers = ["Topic :: Scientific/Engineering :: Artificial Intelligence"]
)