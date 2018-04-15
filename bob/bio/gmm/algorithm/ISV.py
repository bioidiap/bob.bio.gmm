#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.core
import bob.io.base
import bob.learn.em

import numpy
import types

from .GMM import GMM
from bob.bio.base.algorithm import Algorithm

import logging
logger = logging.getLogger("bob.bio.gmm")

class ISV (GMM):
  """Tool for computing Unified Background Models and Gaussian Mixture Models of the features"""


  def __init__(
      self,
      # ISV training
      subspace_dimension_of_u,       # U subspace dimension
      isv_training_iterations = 10,  # Number of EM iterations for the ISV training
      # ISV enrollment
      isv_enroll_iterations = 1,     # Number of iterations for the enrollment phase

      multiple_probe_scoring = None, # scoring when multiple probe files are available

      # parameters of the GMM
      **kwargs
  ):
    """Initializes the local UBM-GMM tool with the given file selector object"""
    # call base class constructor with its set of parameters
    GMM.__init__(self, **kwargs)

    # call tool constructor to overwrite what was set before
    Algorithm.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = False, # not needed anymore because it's done while training the projector
        split_training_features_by_client = True,

        subspace_dimension_of_u = subspace_dimension_of_u,
        isv_training_iterations = isv_training_iterations,
        isv_enroll_iterations = isv_enroll_iterations,

        multiple_model_scoring = None,
        multiple_probe_scoring = multiple_probe_scoring,
        **kwargs
    )

    self.subspace_dimension_of_u = subspace_dimension_of_u
    self.isv_training_iterations = isv_training_iterations
    self.isv_enroll_iterations = isv_enroll_iterations
    self.isv_trainer = bob.learn.em.ISVTrainer(self.relevance_factor)


  def train_isv(self, data):
    """Train the ISV model given a dataset"""
    logger.info("  -> Training ISV enroller")
    self.isvbase = bob.learn.em.ISVBase(self.ubm, self.subspace_dimension_of_u)
    # train ISV model
    # Reseting the pseudo random number generator so we can have the same initialization for serial and parallel execution. 
    self.rng = bob.core.random.mt19937(self.init_seed)
    bob.learn.em.train(self.isv_trainer, self.isvbase, data, self.isv_training_iterations, rng=self.rng)


  def train_projector(self, train_features, projector_file):
    """Train Projector and Enroller at the same time"""
    [self._check_feature(feature) for client in train_features for feature in client]

    data1 = numpy.vstack(feature for client in train_features for feature in client)
    self.train_ubm(data1)
    # to save some memory, we might want to delete these data
    del data1

    # project training data
    logger.info("  -> Projecting training data")
    data = [[self.project_ubm(feature) for feature in client] for client in train_features]

    # train ISV
    self.train_isv(data)

    # Save the ISV base AND the UBM into the same file
    self.save_projector(projector_file)


  def save_projector(self, projector_file):
    """Save the GMM and the ISV model in the same HDF5 file"""
    hdf5file = bob.io.base.HDF5File(projector_file, "w")
    hdf5file.create_group('Projector')
    hdf5file.cd('Projector')
    self.ubm.save(hdf5file)

    hdf5file.cd('/')
    hdf5file.create_group('Enroller')
    hdf5file.cd('Enroller')
    self.isvbase.save(hdf5file)

  def load_isv(self, isv_file):
    hdf5file = bob.io.base.HDF5File(isv_file)
    self.isvbase = bob.learn.em.ISVBase(hdf5file)
    # add UBM model from base class
    self.isvbase.ubm = self.ubm

  def load_projector(self, projector_file):
    """Load the GMM and the ISV model from the same HDF5 file"""
    hdf5file = bob.io.base.HDF5File(projector_file)

    # Load Projector
    hdf5file.cd('/Projector')
    self.load_ubm(hdf5file)

    # Load Enroller
    hdf5file.cd('/Enroller')
    self.load_isv(hdf5file)


  #######################################################
  ################ ISV training #########################
  def project_isv(self, projected_ubm):
    projected_isv = numpy.ndarray(shape=(self.ubm.shape[0]*self.ubm.shape[1],), dtype=numpy.float64)
    model = bob.learn.em.ISVMachine(self.isvbase)
    model.estimate_ux(projected_ubm, projected_isv)
    return projected_isv

  def project(self, feature):
    """Computes GMM statistics against a UBM, then corresponding Ux vector"""
    self._check_feature(feature)
    projected_ubm = GMM.project(self, feature)
    projected_isv = self.project_isv(projected_ubm)
    return [projected_ubm, projected_isv]

  #######################################################
  ################## ISV model enroll ####################

  def write_feature(self, data, feature_file):
    gmmstats = data[0]
    Ux = data[1]
    hdf5file = bob.io.base.HDF5File(feature_file, "w") if isinstance(feature_file, str) else feature_file
    hdf5file.create_group('gmmstats')
    hdf5file.cd('gmmstats')
    gmmstats.save(hdf5file)
    hdf5file.cd('..')
    hdf5file.set('Ux', Ux)

  def read_feature(self, feature_file):
    """Read the type of features that we require, namely GMMStats"""
    hdf5file = bob.io.base.HDF5File(feature_file)
    hdf5file.cd('gmmstats')
    gmmstats = bob.learn.em.GMMStats(hdf5file)
    hdf5file.cd('..')
    Ux = hdf5file.read('Ux')
    return [gmmstats, Ux]


  def _check_projected(self, probe):
    """Checks that the probe is of the desired type"""
    assert isinstance(probe, (tuple, list))
    assert len(probe) == 2
    assert isinstance(probe[0], bob.learn.em.GMMStats)
    assert isinstance(probe[1], numpy.ndarray) and probe[1].ndim == 1 and probe[1].dtype == numpy.float64


  def enroll(self, enroll_features):
    """Performs ISV enrollment"""
    for feature in enroll_features:
      self._check_projected(feature)
    machine = bob.learn.em.ISVMachine(self.isvbase)
    self.isv_trainer.enroll(machine, [f[0] for f in enroll_features], self.isv_enroll_iterations)
    # return the resulting gmm
    return machine


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the ISV Machine that holds the model"""
    machine = bob.learn.em.ISVMachine(bob.io.base.HDF5File(model_file))
    machine.isv_base = self.isvbase
    return machine



  def score(self, model, probe):
    """Computes the score for the given model and the given probe."""
    assert isinstance(model, bob.learn.em.ISVMachine)
    self._check_projected(probe)

    gmmstats = probe[0]
    Ux = probe[1]
    return model.forward_ux(gmmstats, Ux)

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    assert isinstance(model, bob.learn.em.ISVMachine)
    [self._check_projected(probe) for probe in probes]
    if self.probe_fusion_function is not None:
      # When a multiple probe fusion function is selected, use it
      return Algorithm.score_for_multiple_probes(self, model, probes)
    else:
      # Otherwise: compute joint likelihood of all probe features
      # create GMM statistics from first probe statistics
#      import pdb; pdb.set_trace()
      gmmstats_acc = bob.learn.em.GMMStats(probes[0][0])
#      gmmstats_acc = probes[0][0]
      # add all other probe statistics
      for i in range(1,len(probes)):
        gmmstats_acc += probes[i][0]
      # compute ISV score with the accumulated statistics
      projected_isv_acc = numpy.ndarray(shape=(self.ubm.shape[0]*self.ubm.shape[1],), dtype=numpy.float64)
      model.estimate_ux(gmmstats_acc, projected_isv_acc)
      return model.forward_ux(gmmstats_acc, projected_isv_acc)
