#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>


import bob.core
import bob.io.base
import bob.learn.em

import numpy

from bob.bio.base.algorithm import Algorithm
from multiprocessing.pool import ThreadPool

import logging
logger = logging.getLogger("bob.bio.gmm")

class GMM (Algorithm):
  """Algorithm for computing Universal Background Models and Gaussian Mixture Models of the features.
  Features must be normalized to zero mean and unit standard deviation."""

  def __init__(
      self,
      # parameters for the GMM
      number_of_gaussians,
      # parameters of UBM training
      kmeans_training_iterations = 25,   # Maximum number of iterations for K-Means
      gmm_training_iterations = 25,      # Maximum number of iterations for ML GMM Training
      training_threshold = 5e-4,         # Threshold to end the ML training
      variance_threshold = 5e-4,         # Minimum value that a variance can reach
      update_weights = True,
      update_means = True,
      update_variances = True,
      # parameters of the GMM enrollment
      relevance_factor = 4,         # Relevance factor as described in Reynolds paper
      gmm_enroll_iterations = 1,    # Number of iterations for the enrollment phase
      responsibility_threshold = 0, # If set, the weight of a particular Gaussian will at least be greater than this threshold. In the case the real weight is lower, the prior mean value will be used to estimate the current mean and variance.
      INIT_SEED = 5489,
      # scoring
      scoring_function = bob.learn.em.linear_scoring,
      n_threads=None,
  ):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""

    # call base class constructor and register that this tool performs projection
    Algorithm.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = False,

        number_of_gaussians = number_of_gaussians,
        kmeans_training_iterations = kmeans_training_iterations,
        gmm_training_iterations = gmm_training_iterations,
        training_threshold = training_threshold,
        variance_threshold = variance_threshold,
        update_weights = update_weights,
        update_means = update_means,
        update_variances = update_variances,
        relevance_factor = relevance_factor,
        gmm_enroll_iterations = gmm_enroll_iterations,
        responsibility_threshold = responsibility_threshold,
        INIT_SEED = INIT_SEED,
        scoring_function = str(scoring_function),

        multiple_model_scoring = None,
        multiple_probe_scoring = 'average'
    )

    # copy parameters
    self.gaussians = number_of_gaussians
    self.kmeans_training_iterations = kmeans_training_iterations
    self.gmm_training_iterations = gmm_training_iterations
    self.training_threshold = training_threshold
    self.variance_threshold = variance_threshold
    self.update_weights = update_weights
    self.update_means = update_means
    self.update_variances = update_variances
    self.relevance_factor = relevance_factor
    self.gmm_enroll_iterations = gmm_enroll_iterations
    self.init_seed = INIT_SEED
    self.rng = bob.core.random.mt19937(self.init_seed)
    self.responsibility_threshold = responsibility_threshold
    self.scoring_function = scoring_function
    self.n_threads = n_threads
    self.pool = None

    self.ubm = None
    self.kmeans_trainer = bob.learn.em.KMeansTrainer()
    self.ubm_trainer = bob.learn.em.ML_GMMTrainer(self.update_means, self.update_variances, self.update_weights, self.responsibility_threshold)


  def _check_feature(self, feature):
    """Checks that the features are appropriate"""
    if not isinstance(feature, numpy.ndarray) or feature.ndim != 2 or feature.dtype != numpy.float64:
      raise ValueError("The given feature is not appropriate")
    if self.ubm is not None and feature.shape[1] != self.ubm.shape[1]:
      raise ValueError("The given feature is expected to have %d elements, but it has %d" % (self.ubm.shape[1], feature.shape[1]))




  #######################################################
  ################ UBM training #########################

  def train_ubm(self, array):

    logger.debug(" .... Training with %d feature vectors", array.shape[0])
    if self.n_threads is not None:
        self.pool = ThreadPool(self.n_threads)

    # Computes input size
    input_size = array.shape[1]

    # Creates the machines (KMeans and GMM)
    logger.debug(" .... Creating machines")
    kmeans = bob.learn.em.KMeansMachine(self.gaussians, input_size)
    self.ubm = bob.learn.em.GMMMachine(self.gaussians, input_size)

    # Trains using the KMeansTrainer
    logger.info("  -> Training K-Means")

    # Reseting the pseudo random number generator so we can have the same initialization for serial and parallel execution.
    self.rng = bob.core.random.mt19937(self.init_seed)
    bob.learn.em.train(
        self.kmeans_trainer, kmeans, array, self.kmeans_training_iterations,
        self.training_threshold, rng=self.rng, pool=self.pool,
    )

    variances, weights = kmeans.get_variances_and_weights_for_each_cluster(array)
    means = kmeans.means

    # Initializes the GMM
    self.ubm.means = means
    self.ubm.variances = variances
    self.ubm.weights = weights
    self.ubm.set_variance_thresholds(self.variance_threshold)

    # Trains the GMM
    logger.info("  -> Training GMM")
    # Reseting the pseudo random number generator so we can have the same initialization for serial and parallel execution.
    self.rng = bob.core.random.mt19937(self.init_seed)
    bob.learn.em.train(
        self.ubm_trainer, self.ubm, array, self.gmm_training_iterations,
        self.training_threshold, rng=self.rng, pool=self.pool,
    )

  def save_ubm(self, projector_file):
    """Save projector to file"""
    # Saves the UBM to file
    logger.debug(" .... Saving model to file '%s'", projector_file)
    hdf5 = projector_file if isinstance(projector_file, bob.io.base.HDF5File) else bob.io.base.HDF5File(projector_file, 'w')
    self.ubm.save(hdf5)

  def train_projector(self, train_features, projector_file):
    """Computes the Universal Background Model from the training ("world") data"""
    [self._check_feature(feature) for feature in train_features]

    logger.info("  -> Training UBM model with %d training files", len(train_features))

    # Loads the data into an array
    array = numpy.vstack(train_features)

    self.train_ubm(array)

    self.save_ubm(projector_file)

  #######################################################
  ############## GMM training using UBM #################

  def load_ubm(self, ubm_file):
    hdf5file = bob.io.base.HDF5File(ubm_file)
    # read UBM
    self.ubm = bob.learn.em.GMMMachine(hdf5file)
    self.ubm.set_variance_thresholds(self.variance_threshold)


  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
    # read UBM
    self.load_ubm(projector_file)
    # prepare MAP_GMM_Trainer
    kwargs = dict(mean_var_update_responsibilities_threshold=self.responsibility_threshold) if self.responsibility_threshold > 0. else dict()
    self.enroll_trainer = bob.learn.em.MAP_GMMTrainer(self.ubm, relevance_factor = self.relevance_factor, update_means = True, update_variances = False, **kwargs)
    self.rng = bob.core.random.mt19937(self.init_seed)


  def project_ubm(self, array):
    logger.debug(" .... Projecting %d feature vectors" % array.shape[0])
    # Accumulates statistics
    gmm_stats = bob.learn.em.GMMStats(self.ubm.shape[0], self.ubm.shape[1])
    self.ubm.acc_statistics(array, gmm_stats)

    # return the resulting statistics
    return gmm_stats


  def project(self, feature):
    """Computes GMM statistics against a UBM, given an input 2D numpy.ndarray of feature vectors"""
    self._check_feature(feature)
    return self.project_ubm(feature)


  def read_gmm_stats(self, gmm_stats_file):
    """Reads GMM stats from file."""
    return bob.learn.em.GMMStats(bob.io.base.HDF5File(gmm_stats_file))

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely GMM_Stats"""
    return self.read_gmm_stats(feature_file)

  def enroll_gmm(self, array):
    logger.debug(" .... Enrolling with %d feature vectors", array.shape[0])

    gmm = bob.learn.em.GMMMachine(self.ubm)
    gmm.set_variance_thresholds(self.variance_threshold)
    bob.learn.em.train(
        self.enroll_trainer, gmm, array, self.gmm_enroll_iterations,
        self.training_threshold, rng=self.rng, pool=self.pool,
    )
    return gmm

  def enroll(self, feature_arrays):
    """Enrolls a GMM using MAP adaptation, given a list of 2D numpy.ndarray's of feature vectors"""
    [self._check_feature(feature) for feature in feature_arrays]
    array = numpy.vstack(feature_arrays)
    # Use the array to train a GMM and return it
    return self.enroll_gmm(array)


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the model, which is a GMM machine"""
    return bob.learn.em.GMMMachine(bob.io.base.HDF5File(model_file))

  def score(self, model, probe):
    """Computes the score for the given model and the given probe using the scoring function from the config file"""
    assert isinstance(model, bob.learn.em.GMMMachine)
    assert isinstance(probe, bob.learn.em.GMMStats)
    return self.scoring_function([model], self.ubm, [probe], [], frame_length_normalisation = True)[0][0]

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    assert isinstance(model, bob.learn.em.GMMMachine)
    for probe in probes:
      assert isinstance(probe, bob.learn.em.GMMStats)
#    logger.warn("Please verify that this function is correct")
    return self.probe_fusion_function(self.scoring_function([model], self.ubm, probes, [], frame_length_normalisation = True))





class GMMRegular (GMM):
  """Algorithm for computing Universal Background Models and Gaussian Mixture Models of the features"""

  def __init__(self, **kwargs):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""
#    logger.warn("This class must be checked. Please verify that I didn't do any mistake here. I had to rename 'train_projector' into a 'train_enroller'!")
    # initialize the UBMGMM base class
    GMM.__init__(self, **kwargs)
    # register a different set of functions in the Tool base class
    Algorithm.__init__(self, requires_enroller_training = True, performs_projection = False)


  #######################################################
  ################ UBM training #########################

  def train_enroller(self, train_features, enroller_file):
    """Computes the Universal Background Model from the training ("world") data"""
    train_features = [feature for client in train_features for feature in client]
    return self.train_projector(train_features, enroller_file)


  #######################################################
  ############## GMM training using UBM #################

  def load_enroller(self, enroller_file):
    """Reads the UBM model from file"""
    return self.load_projector(enroller_file)


  ######################################################
  ################ Feature comparison ##################
  def score(self, model, probe):
    """Computes the score for the given model and the given probe.
    The score are Log-Likelihood.
    Therefore, the log of the likelihood ratio is obtained by computing the following difference."""

    assert isinstance(model, bob.learn.em.GMMMachine)
    self._check_feature(probe)
    score = sum(model.log_likelihood(probe[i,:]) - self.ubm.log_likelihood(probe[i,:]) for i in range(probe.shape[0]))
    return score/probe.shape[0]

  def score_for_multiple_probes(self, model, probes):
    raise NotImplementedError("Implement Me!")
