#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import bob.core
import bob.io.base
import bob.learn.em


from .GMM import GMM
from bob.bio.base.algorithm import Algorithm

import logging
logger = logging.getLogger("bob.bio.gmm")

class JFA (GMM):
  """Tool for computing Unified Background Models and Gaussian Mixture Models of the features and project it via JFA"""

  def __init__(
      self,
      # JFA training
      subspace_dimension_of_u,       # U subspace dimension
      subspace_dimension_of_v,       # V subspace dimension
      jfa_training_iterations = 10,  # Number of EM iterations for the JFA training
      # JFA enrollment
      jfa_enroll_iterations = 1,     # Number of iterations for the enrollment phase
      # parameters of the GMM
      **kwargs
  ):
    """Initializes the local UBM-GMM tool with the given file selector object"""
    # call base class constructor
    GMM.__init__(self, **kwargs)

    # call tool constructor to overwrite what was set before
    Algorithm.__init__(
        self,
        performs_projection = True,
        use_projected_features_for_enrollment = True,
        requires_enroller_training = True,

        subspace_dimension_of_u = subspace_dimension_of_u,
        subspace_dimension_of_v = subspace_dimension_of_v,
        jfa_training_iterations = jfa_training_iterations,
        jfa_enroll_iterations = jfa_enroll_iterations,

        multiple_model_scoring = None,
        multiple_probe_scoring = None,
        **kwargs
    )

    self.subspace_dimension_of_u = subspace_dimension_of_u
    self.subspace_dimension_of_v = subspace_dimension_of_v
    self.jfa_training_iterations = jfa_training_iterations
    self.jfa_enroll_iterations = jfa_enroll_iterations
    self.jfa_trainer = bob.learn.em.JFATrainer()


  def load_projector(self, projector_file):
    """Reads the UBM model from file"""
    # Here, we just need to load the UBM from the projector file.
    self.load_ubm(projector_file)


  #######################################################
  ################ JFA training #########################
  def train_enroller(self, train_features, enroller_file):
    # assert that all training features are GMMStatistics
    for client_feature in train_features:
      for feature in client_feature:
        assert isinstance(feature, bob.learn.em.GMMStats)

    # create a JFABasemachine with the UBM from the base class
    self.jfa_base = bob.learn.em.JFABase(self.ubm, self.subspace_dimension_of_u, self.subspace_dimension_of_v)

    # train the JFA
    bob.learn.em.train_jfa(self.jfa_trainer, self.jfa_base, train_features, self.jfa_training_iterations, rng = bob.core.random.mt19937(self.init_seed))

    # Save the JFA base AND the UBM into the same file
    self.jfa_base.save(bob.io.base.HDF5File(enroller_file, "w"))



  #######################################################
  ################## JFA model enroll ####################
  def load_enroller(self, enroller_file):
    """Reads the JFA base from file"""
    # now, load the JFA base, if it is included in the file
    self.jfa_base = bob.learn.em.JFABase(bob.io.base.HDF5File(enroller_file))
    # add UBM model from base class
    self.jfa_base.ubm = self.ubm

    # TODO: Why is the rng re-initialized here?
    #self.rng = bob.core.random.mt19937(self.init_seed)


  def read_feature(self, feature_file):
    """Reads the projected feature to be enrolled as a model"""
    return bob.learn.em.GMMStats(bob.io.base.HDF5File(feature_file))


  def enroll(self, enroll_features):
    """Enrolls a GMM using MAP adaptation"""
    machine = bob.learn.em.JFAMachine(self.jfa_base)
    self.jfa_trainer.enroll(machine, enroll_features, self.jfa_enroll_iterations)
    # return the resulting gmm
    return machine


  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the JFA Machine that holds the model"""
    machine = bob.learn.em.JFAMachine(bob.io.base.HDF5File(model_file))
    machine.jfa_base = self.jfa_base
    return machine

  def score(self, model, probe):
    """Computes the score for the given model and the given probe"""
    assert isinstance(model, bob.learn.em.JFAMachine)
    assert isinstance(probe, bob.learn.em.GMMStats)
    return model.log_likelihood(probe)

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several probes."""
    # TODO: Check if this is correct
#    logger.warn("This function needs to be verified!")
    raise NotImplementedError('Multiple probes is not yet supported')
    #scores = numpy.ndarray((len(probes),), 'float64')
    #model.forward(probes, scores)
    #return scores[0]
