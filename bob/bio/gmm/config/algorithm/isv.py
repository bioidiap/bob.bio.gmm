import bob.bio.gmm

algorithm = bob.bio.gmm.algorithm.ISV(
    # ISV parameters
    subspace_dimension_of_u = 160,
    # GMM parameters
    number_of_gaussians = 512
)
