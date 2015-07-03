import bob.bio.gmm

algorithm = bob.bio.gmm.algorithm.IVector(
    # IVector parameters
    subspace_dimension_of_t = 100,
    update_sigma = True,
    tv_training_iterations = 25,  # Number of EM iterations for the TV training
    # GMM parameters
    number_of_gaussians = 256,
    use_plda = True,
    plda_dim_F  = 50,
    plda_dim_G = 50,
    plda_training_iterations = 200,
)
