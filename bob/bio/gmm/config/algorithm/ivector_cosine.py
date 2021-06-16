import bob.bio.gmm

algorithm = bob.bio.gmm.algorithm.IVector(
    # IVector parameters
    subspace_dimension_of_t=400,
    update_sigma=True,
    tv_training_iterations=3,  # Number of EM iterations for the TV training
    # GMM parameters
    number_of_gaussians=512,
)
