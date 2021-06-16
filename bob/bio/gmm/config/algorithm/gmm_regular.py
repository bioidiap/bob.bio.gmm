#!/usr/bin/env python

import numpy

import bob.bio.gmm

algorithm = bob.bio.gmm.algorithm.GMMRegular(number_of_gaussians=512)
