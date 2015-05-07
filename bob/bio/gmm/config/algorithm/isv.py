#!/usr/bin/env python

import bob.bio.gmm
import numpy

algorithm = bob.bio.gmm.algorithm.ISV(
    # ISV parameters
    subspace_dimension_of_u = 160,
    # GMM parameters
    number_of_gaussians = 512,
    # by default, our features are normalized, so it does not need to be done here
    normalize_before_k_means = False
)
