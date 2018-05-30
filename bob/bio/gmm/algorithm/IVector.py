#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

import bob.core
import bob.io.base
import bob.learn.linear
import bob.learn.em

import numpy

from .GMM import GMM
from bob.bio.base.algorithm import Algorithm

import logging
logger = logging.getLogger("bob.bio.gmm")

class IVector (GMM):
  """Tool for extracting I-Vectors"""

  def __init__(
      self,
      # IVector training
      subspace_dimension_of_t,       # T subspace dimension
      tv_training_iterations = 25,   # Number of EM iterations for the JFA training
      update_sigma = True,
      use_whitening = True,
      use_lda = False,
      use_wccn = False,
      use_plda = False,
      lda_dim = None,
      lda_strip_to_rank=True,
      plda_dim_F  = 50,
      plda_dim_G = 50,
      plda_training_iterations = 50,
      # parameters of the GMM
      **kwargs
  ):
    """Initializes the local GMM tool with the given file selector object"""
    # call base class constructor with its set of parameters
    GMM.__init__(self, **kwargs)

    # call tool constructor to overwrite what was set before
    Algorithm.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = False, # not needed anymore because it's done while training the projector
        split_training_features_by_client = True,

        subspace_dimension_of_t = subspace_dimension_of_t,
        tv_training_iterations = tv_training_iterations,
        update_sigma = update_sigma,
        use_whitening = use_whitening,
        use_lda = use_lda,
        use_wccn = use_wccn,
        use_plda = use_plda,
        lda_dim = lda_dim,
        lda_strip_to_rank = lda_strip_to_rank,
        plda_dim_F  = plda_dim_F,
        plda_dim_G = plda_dim_G,
        plda_training_iterations = plda_training_iterations,

        multiple_model_scoring = None,
        multiple_probe_scoring = None,
        **kwargs
    )

    self.update_sigma = update_sigma
    self.use_whitening = use_whitening
    self.use_lda = use_lda
    self.use_wccn = use_wccn
    self.use_plda = use_plda
    self.subspace_dimension_of_t = subspace_dimension_of_t
    self.tv_training_iterations = tv_training_iterations

    self.ivector_trainer = bob.learn.em.IVectorTrainer(update_sigma=update_sigma)
    self.whitening_trainer = bob.learn.linear.WhiteningTrainer()

    self.lda_dim = lda_dim
    self.lda_trainer = bob.learn.linear.FisherLDATrainer(strip_to_rank=lda_strip_to_rank)
    self.wccn_trainer = bob.learn.linear.WCCNTrainer()
    self.plda_trainer = bob.learn.em.PLDATrainer()
    self.plda_dim_F  = plda_dim_F
    self.plda_dim_G = plda_dim_G
    self.plda_training_iterations = plda_training_iterations



  def _check_ivector(self, feature):
    """Checks that the features are appropriate"""
    if not isinstance(feature, numpy.ndarray) or feature.ndim != 1 or feature.dtype != numpy.float64:
      raise ValueError("The given feature is not appropriate")

  def train_ivector(self, training_stats):
    logger.info("  -> Training IVector enroller")
    self.tv = bob.learn.em.IVectorMachine(self.ubm, self.subspace_dimension_of_t, self.variance_threshold)

    # Reseting the pseudo random number generator so we can have the same initialization for serial and parallel execution. 
    self.rng = bob.core.random.mt19937(self.init_seed)

    # train IVector model
    bob.learn.em.train(self.ivector_trainer, self.tv, training_stats, self.tv_training_iterations, rng=self.rng)


  def train_whitener(self, training_features):
    logger.info("  -> Training Whitening")
    ivectors_matrix = numpy.vstack(training_features)
    # create a Linear Machine
    self.whitener = bob.learn.linear.Machine(ivectors_matrix.shape[1],ivectors_matrix.shape[1])
    # create the whitening trainer
    self.whitening_trainer.train(ivectors_matrix, self.whitener)

  def train_lda(self, training_features):
    logger.info("  -> Training LDA projector")
    self.lda, __eig_vals = self.lda_trainer.train(training_features)

    # resize the machine if desired
    # You can only clip if the rank is higher than LDA_DIM 
    if self.lda_dim is not None:
      if len(__eig_vals) < self.lda_dim:
        logger.warning("  -> You are resizing the LDA matrix to a value above its rank"
                       "(from {0} to {1}). Be aware that this may lead you to imprecise eigenvectors.".\
                        format(len(__eig_vals), self.lda_dim))
      self.lda.resize(self.lda.shape[0], self.lda_dim)
       

  def train_wccn(self, training_features):
    logger.info("  -> Training WCCN projector")
    self.wccn = self.wccn_trainer.train(training_features)

  def train_plda(self, training_features):
    logger.info("  -> Training PLDA projector")
    self.plda_trainer.init_f_method = 'BETWEEN_SCATTER'
    self.plda_trainer.init_g_method = 'WITHIN_SCATTER'
    self.plda_trainer.init_sigma_method = 'VARIANCE_DATA'
    variance_flooring = 1e-5
    training_features = [numpy.vstack(client) for client in training_features]
    input_dim = training_features[0].shape[1]

    # Reseting the pseudo random number generator so we can have the same initialization for serial and parallel execution. 
    self.rng = bob.core.random.mt19937(self.init_seed)
    
    self.plda_base = bob.learn.em.PLDABase(input_dim, self.plda_dim_F, self.plda_dim_G, variance_flooring)
    bob.learn.em.train(self.plda_trainer, self.plda_base, training_features, self.plda_training_iterations, rng=self.rng)


  def train_projector(self, train_features, projector_file):
    """Train Projector and Enroller at the same time"""

    [self._check_feature(feature) for client in train_features for feature in client]

    # train UBM
    data = numpy.vstack(feature for client in train_features for feature in client)
    self.train_ubm(data)
    del data

    # project training data
    logger.info("  -> Projecting training data")
    train_gmm_stats = [[self.project_ubm(feature) for feature in client] for client in train_features]
    train_gmm_stats_flatten = [stats for client in train_gmm_stats for stats in client]

    # train IVector
    logger.info("  -> Projecting training data")
    self.train_ivector(train_gmm_stats_flatten)

    # project training i-vectors
    train_ivectors = [[self.project_ivector(stats) for stats in client] for client in train_gmm_stats]
    train_ivectors_flatten = [stats for client in train_ivectors for stats in client]

    if self.use_whitening:
      # Train Whitening
      self.train_whitener(train_ivectors_flatten)
      # whitening and length-normalizing i-vectors
      train_ivectors = [[self.project_whitening(ivec) for ivec in client] for client in train_ivectors]

    if self.use_lda:
      self.train_lda(train_ivectors)
      train_ivectors = [[self.project_lda(ivec) for ivec in client] for client in train_ivectors]

    if self.use_wccn:
      self.train_wccn(train_ivectors)
      train_ivectors = [[self.project_wccn(ivec) for ivec in client] for client in train_ivectors]

    if self.use_plda:
      self.train_plda(train_ivectors)

    # save
    self.save_projector(projector_file)


  def save_projector(self, projector_file):
    # Save the IVector base AND the UBM AND the whitening into the same file
    hdf5file = bob.io.base.HDF5File(projector_file, "w")
    hdf5file.create_group('Projector')
    hdf5file.cd('Projector')
    self.save_ubm(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Enroller')
    hdf5file.cd('Enroller')
    self.tv.save(hdf5file)

    if self.use_whitening:
      hdf5file.cd('/')
      hdf5file.create_group('Whitener')
      hdf5file.cd('Whitener')
      self.whitener.save(hdf5file)

    if self.use_lda:
      hdf5file.cd('/')
      hdf5file.create_group('LDA')
      hdf5file.cd('LDA')
      self.lda.save(hdf5file)

    if self.use_wccn:
      hdf5file.cd('/')
      hdf5file.create_group('WCCN')
      hdf5file.cd('WCCN')
      self.wccn.save(hdf5file)

    if self.use_plda:
      hdf5file.cd('/')
      hdf5file.create_group('PLDA')
      hdf5file.cd('PLDA')
      self.plda_base.save(hdf5file)


  def load_tv(self, tv_file):
    hdf5file = bob.io.base.HDF5File(tv_file)
    self.tv = bob.learn.em.IVectorMachine(hdf5file)
    # add UBM model from base class
    self.tv.ubm = self.ubm

  def load_whitener(self, whitening_file):
    hdf5file = bob.io.base.HDF5File(whitening_file)
    self.whitener = bob.learn.linear.Machine(hdf5file)

  def load_lda(self, lda_file):
    hdf5file = bob.io.base.HDF5File(lda_file)
    self.lda = bob.learn.linear.Machine(hdf5file)

  def load_wccn(self, wccn_file):
    hdf5file = bob.io.base.HDF5File(wccn_file)
    self.wccn = bob.learn.linear.Machine(hdf5file)

  def load_plda(self, plda_file):
    hdf5file = bob.io.base.HDF5File(plda_file)
    self.plda_base = bob.learn.em.PLDABase(hdf5file)
    self.plda_machine = bob.learn.em.PLDAMachine(self.plda_base)

  def load_projector(self, projector_file):
    """Load the GMM and the ISV model from the same HDF5 file"""
    hdf5file = bob.io.base.HDF5File(projector_file)

    # Load Projector
    hdf5file.cd('/Projector')
    self.load_ubm(hdf5file)

    # Load Enroller
    hdf5file.cd('/Enroller')
    self.load_tv(hdf5file)

    if self.use_whitening:
      # Load Whitening
      hdf5file.cd('/Whitener')
      self.load_whitener(hdf5file)

    if self.use_lda:
      # Load LDA
      hdf5file.cd('/LDA')
      self.load_lda(hdf5file)

    if self.use_wccn:
      # Load WCCN
      hdf5file.cd('/WCCN')
      self.load_wccn(hdf5file)

    if self.use_plda:
     # Load PLDA
      hdf5file.cd('/PLDA')
      self.load_plda(hdf5file)


  def project_ivector(self, gmm_stats):
    return self.tv.project(gmm_stats)

  def project_whitening(self, ivector):
    whitened = self.whitener.forward(ivector)
    return whitened / numpy.linalg.norm(whitened)

  def project_lda(self, ivector):
    out_ivector = numpy.ndarray(self.lda.shape[1], numpy.float64)
    self.lda(ivector, out_ivector)
    return out_ivector

  def project_wccn(self, ivector):
    out_ivector = numpy.ndarray(self.wccn.shape[1], numpy.float64)
    self.wccn(ivector, out_ivector)
    return out_ivector

  #######################################################
  ############## IVector projection #####################
  def project(self, feature_array):
    """Computes GMM statistics against a UBM, then corresponding Ux vector"""
    self._check_feature(feature_array)
    # project UBM
    projected_ubm = self.project_ubm(feature_array)
    # project I-Vector
    ivector = self.project_ivector(projected_ubm)
    # whiten I-Vector
    if self.use_whitening:
      ivector = self.project_whitening(ivector)
    # LDA projection
    if self.use_lda:
      ivector = self.project_lda(ivector)
    # WCCN projection
    if self.use_wccn:
      ivector = self.project_wccn(ivector)
    return ivector

  #######################################################
  ################## Read / Write I-Vectors ####################
  def write_feature(self, data, feature_file):
    """Saves the feature, which is the (whitened) I-Vector."""
    bob.bio.base.save(data, feature_file)

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely i-vectors (stored as simple numpy arrays)"""
    return bob.bio.base.load(feature_file)


  #######################################################
  ################## Model  Enrollment ###################
  def enroll(self, enroll_features):
    """Performs IVector enrollment"""
    [self._check_ivector(feature) for feature in enroll_features]
    average_ivector = numpy.mean(numpy.vstack(enroll_features), axis=0)
    if self.use_plda:
      average_ivector = average_ivector.reshape(1,-1)
      self.plda_trainer.enroll(self.plda_machine, average_ivector)
      return self.plda_machine
    else:
      return average_ivector


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the whitened i-vector that holds the model"""
    if self.use_plda:
      return bob.learn.em.PLDAMachine(bob.io.base.HDF5File(str(model_file)), self.plda_base)
    else:
      return bob.bio.base.load(model_file)


  def score(self, model, probe):
    """Computes the score for the given model and the given probe."""
    self._check_ivector(probe)
    if self.use_plda:
      return model.log_likelihood_ratio(probe)
    else:
      self._check_ivector(model)
      return numpy.dot(model/numpy.linalg.norm(model), probe/numpy.linalg.norm(probe))


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    probe = numpy.mean(numpy.vstack(probes), axis=0)
    return self.score(model, probe)
