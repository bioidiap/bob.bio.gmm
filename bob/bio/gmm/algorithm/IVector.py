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
        split_training_features_by_client = False,

        subspace_dimension_of_t = subspace_dimension_of_t,
        tv_training_iterations = tv_training_iterations,
        update_sigma = update_sigma,

        multiple_model_scoring = None,
        multiple_probe_scoring = None,
        **kwargs
    )

    self.update_sigma = update_sigma
    self.subspace_dimension_of_t = subspace_dimension_of_t
    self.tv_training_iterations = tv_training_iterations
    self.ivector_trainer = bob.learn.em.IVectorTrainer(update_sigma=update_sigma)
    self.whitening_trainer = bob.learn.linear.WhiteningTrainer()


  def _check_projected(self, feature):
    """Checks that the features are appropriate"""
    if not isinstance(feature, numpy.ndarray) or feature.ndim != 1 or feature.dtype != numpy.float64:
      raise ValueError("The given feature is not appropriate")
    if self.whitener is not None and feature.shape[0] != self.whitener.shape[1]:
      raise ValueError("The given feature is expected to have %d elements, but it has %d" % (self.whitener.shape[1], feature.shape[0]))


  def train_ivector(self, training_stats):
    logger.info("  -> Training IVector enroller")
    self.tv = bob.learn.em.IVectorMachine(self.ubm, self.subspace_dimension_of_t, self.variance_threshold)

    # train IVector model
    bob.learn.em.train(self.ivector_trainer, self.tv, training_stats, self.tv_training_iterations, rng=self.rng)


  def train_whitener(self, training_features):
    ivectors_matrix = numpy.vstack(training_features)
    # create a Linear Machine
    self.whitener = bob.learn.linear.Machine(ivectors_matrix.shape[1],ivectors_matrix.shape[1])
    # create the whitening trainer
    self.whitening_trainer.train(ivectors_matrix, self.whitener)


  def train_projector(self, train_features, projector_file):
    """Train Projector and Enroller at the same time"""
    [self._check_feature(feature) for feature in train_features]

    # train UBM
    data = numpy.vstack(train_features)
    self.train_ubm(data)
    del data

    # train IVector
    logger.info("  -> Projecting training data")
    training_stats = [self.project_ubm(feature) for feature in train_features]
    # train IVector
    self.train_ivector(training_stats)

    # project training i-vectors
    whitening_train_data = [self.project_ivector(stats) for stats in training_stats]
    self.train_whitener(whitening_train_data)

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

    hdf5file.cd('/')
    hdf5file.create_group('Whitener')
    hdf5file.cd('Whitener')
    self.whitener.save(hdf5file)


  def load_tv(self, tv_file):
    hdf5file = bob.io.base.HDF5File(tv_file)
    self.tv = bob.learn.em.IVectorMachine(hdf5file)
    # add UBM model from base class
    self.tv.ubm = self.ubm

  def load_whitener(self, whitening_file):
    hdf5file = bob.io.base.HDF5File(whitening_file)
    self.whitener = bob.learn.linear.Machine(hdf5file)


  def load_projector(self, projector_file):
    """Load the GMM and the ISV model from the same HDF5 file"""
    hdf5file = bob.io.base.HDF5File(projector_file)

    # Load Projector
    hdf5file.cd('/Projector')
    self.load_ubm(hdf5file)

    # Load Enroller
    hdf5file.cd('/Enroller')
    self.load_tv(hdf5file)

    # Load Whitening
    hdf5file.cd('/Whitener')
    self.load_whitener(hdf5file)


  def project_ivector(self, gmm_stats):
    return self.tv.project(gmm_stats)

  def project_whitening(self, ivector):
    whitened = self.whitener.forward(ivector)
    return whitened / numpy.linalg.norm(whitened)

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
    return self.project_whitening(ivector)

  #######################################################
  ################## ISV model enroll ####################
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
    [self._check_projected(feature) for feature in enroll_features]
    model = numpy.mean(numpy.vstack(enroll_features), axis=0)
    return model


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the whitened i-vector that holds the model"""
    return bob.bio.base.load(model_file)

  def read_probe(self, probe_file):
    """read probe file which is an i-vector"""
    return bob.bio.base.load(probe_file)

  def score(self, model, probe):
    """Computes the score for the given model and the given probe."""
    self._check_projected(model)
    self._check_projected(probe)
    return numpy.dot(model/numpy.linalg.norm(model), probe/numpy.linalg.norm(probe))


  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    [self._check_projected(probe) for probe in probes]
    probe = numpy.mean(numpy.vstack(probes), axis=0)
    return self.score(model, probe)
