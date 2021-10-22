#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Interface between the lower level GMM classes and the Algorithm Transformer.

Implements the enroll and score methods using the low level GMM implementation.

This adds the notions of models, probes, enrollment, and scores to GMM.
"""


import logging

from typing import Callable

import numpy

from sklearn.base import BaseEstimator

import bob.core
import bob.io.base

from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import BioAlgorithm
from bob.learn.em.mixture import GMMMachine
from bob.learn.em.mixture import GMMStats
from bob.learn.em.mixture import linear_scoring

logger = logging.getLogger(__name__)


class GMM(BioAlgorithm, BaseEstimator):
    """Algorithm for computing UBM and Gaussian Mixture Models of the features.

    Features must be normalized to zero mean and unit standard deviation.

    Models are MAP GMM machines trained from a UBM on the enrollment feature set.

    The UBM is a ML GMM machine trained on the training feature set.

    Probes are GMM statistics of features projected on the UBM.
    """

    def __init__(
        self,
        # parameters for the GMM
        number_of_gaussians: int,
        # parameters of UBM training
        kmeans_training_iterations: int = 25,  # Maximum number of iterations for K-Means
        ubm_training_iterations: int = 25,  # Maximum number of iterations for GMM Training
        training_threshold: float = 5e-4,  # Threshold to end the ML training
        variance_threshold: float = 5e-4,  # Minimum value that a variance can reach
        update_weights: bool = True,
        update_means: bool = True,
        update_variances: bool = True,
        # parameters of the GMM enrollment
        relevance_factor: float = 4,  # Relevance factor as described in Reynolds paper
        gmm_enroll_iterations: int = 1,  # Number of iterations for the enrollment phase
        responsibility_threshold: float = 0,  # If set, the weight of a particular Gaussian will at least be greater than this threshold. In the case the real weight is lower, the prior mean value will be used to estimate the current mean and variance.
        init_seed: int = 5489,
        # scoring
        scoring_function: Callable = linear_scoring,
        # n_threads=None,
    ):
        """Initializes the local UBM-GMM tool chain.

        Parameters
        ----------
        number_of_gaussians
            The number of Gaussians used in the UBM and the models.
        kmeans_training_iterations
            Number of e-m iterations to train k-means initializing the UBM.
        ubm_training_iterations
            Number of e-m iterations for training the UBM.
        training_threshold
            Convergence threshold to halt the GMM training early.
        variance_threshold
            Minimum value a variance of the Gaussians can reach.
        update_weights
            Decides wether the weights of the Gaussians are updated while training.
        update_means
            Decides wether the means of the Gaussians are updated while training.
        update_variances
            Decides wether the variancess of the Gaussians are updated while training.
        relevance_factor
            Relevance factor as described in Reynolds paper.
        gmm_enroll_iterations
            Number of iterations for the MAP GMM used for enrollment.
        responsibility_threshold
            If set, the weight of a particular Gaussian will at least be greater than
            this threshold. In the case where the real weight is lower, the prior mean
            value will be used to estimate the current mean and variance.
        init_seed
            Seed for the random number generation.
        scoring_function
            Function returning a score from a model, a UBM, and a probe.
        """

        # call base class constructor and register that this tool performs projection
        # super().__init__(score_reduction_operation=??)

        # copy parameters
        self.number_of_gaussians = number_of_gaussians
        self.kmeans_training_iterations = kmeans_training_iterations
        self.ubm_training_iterations = ubm_training_iterations
        self.training_threshold = training_threshold
        self.variance_threshold = variance_threshold
        self.update_weights = update_weights
        self.update_means = update_means
        self.update_variances = update_variances
        self.relevance_factor = relevance_factor
        self.gmm_enroll_iterations = gmm_enroll_iterations
        self.init_seed = init_seed
        self.rng = bob.core.random.mt19937(self.init_seed)  # TODO
        self.responsibility_threshold = responsibility_threshold
        self.scoring_function = scoring_function

        self.ubm = None

    def _check_feature(self, feature):
        """Checks that the features are appropriate"""
        if (
            not isinstance(feature, numpy.ndarray)
            or feature.ndim != 2
            or feature.dtype != numpy.float64
        ):
            raise ValueError("The given feature is not appropriate")
        if self.ubm is not None and feature.shape[1] != self.ubm.shape[1]:
            raise ValueError(
                "The given feature is expected to have %d elements, but it has %d"
                % (self.ubm.shape[1], feature.shape[1])
            )

    #######################################################
    #                UBM training                         #

    def train_ubm(self, array):

        logger.debug(" .... Training UBM with %d feature vectors", array.shape[0])

        logger.debug(" .... Creating UBM machine")
        self.ubm = GMMMachine(
            n_gaussians=self.number_of_gaussians,
            trainer="ml",
            max_fitting_steps=self.ubm_training_iterations,
            convergence_threshold=self.training_threshold,
            update_means=self.update_means,
            update_variances=self.update_variances,
            update_weights=self.update_weights,
            # TODO more params?
        )

        # Trains the GMM
        logger.info("  -> Training UBM GMM")
        # Resetting the pseudo random number generator so we can have the same initialization for serial and parallel execution.
        # self.rng = bob.core.random.mt19937(self.init_seed)
        self.ubm.fit(array)

    def save_ubm(self, projector_file):
        """Saves the projector to file"""
        # Saves the UBM to file
        logger.debug(" .... Saving model to file '%s'", projector_file)

        hdf5 = (
            projector_file
            if isinstance(projector_file, bob.io.base.HDF5File)
            else bob.io.base.HDF5File(projector_file, "w")
        )
        self.ubm.save(hdf5)

    def train_projector(self, train_features, projector_file):
        """Computes the Universal Background Model from the training ("world") data"""
        [self._check_feature(feature) for feature in train_features]

        logger.info(
            "  -> Training UBM model with %d training files", len(train_features)
        )

        # Loads the data into an array
        array = numpy.vstack(train_features)

        self.train_ubm(array)

        self.save_ubm(projector_file)

    #######################################################
    #              GMM training using UBM                 #

    def load_ubm(self, ubm_file):
        hdf5file = bob.io.base.HDF5File(ubm_file)
        # read UBM
        self.ubm = GMMMachine.from_hdf5(hdf5file)
        self.ubm.variance_thresholds = self.variance_threshold

    def load_projector(self, projector_file):
        """Reads the UBM model from file"""
        # read UBM
        self.load_ubm(projector_file)
        # prepare MAP_GMM_Trainer
        # kwargs = (
        #     dict(
        #         mean_var_update_responsibilities_threshold=self.responsibility_threshold
        #     )
        #     if self.responsibility_threshold > 0.0
        #     else dict()
        # )
        # self.enroll_trainer = bob.learn.em.MAP_GMMTrainer(
        #     self.ubm,
        #     relevance_factor=self.relevance_factor,
        #     update_means=True,
        #     update_variances=False,
        #     **kwargs
        # )
        self.rng = bob.core.random.mt19937(self.init_seed)

    def project_ubm(self, array):
        logger.debug(" .... Projecting %d feature vectors", array.shape[0])
        # Accumulates statistics
        gmm_stats = GMMStats(self.ubm.shape[0], self.ubm.shape[1])
        self.ubm.acc_statistics(array, gmm_stats)

        # return the resulting statistics
        return gmm_stats

    def project(self, feature):
        """Computes GMM statistics against a UBM, given an input 2D numpy.ndarray of feature vectors"""
        self._check_feature(feature)
        return self.project_ubm(feature)

    def read_gmm_stats(self, gmm_stats_file):
        """Reads GMM stats from file."""
        return GMMStats.from_hdf5(bob.io.base.HDF5File(gmm_stats_file))

    def read_feature(self, feature_file):
        """Read the type of features that we require, namely GMM_Stats"""
        return self.read_gmm_stats(feature_file)

    def write_feature(self, feature, feature_file):
        """Write the features (GMM_Stats)"""
        return feature.save(feature_file)

    def enroll_gmm(self, array):
        logger.debug(" .... Enrolling with %d feature vectors", array.shape[0])
        # TODO responsibility_threshold
        gmm = GMMMachine(
            n_gaussians=self.number_of_gaussians,
            trainer="map",
            ubm=self.ubm,
            convergence_threshold=self.training_threshold,
            max_fitting_steps=self.gmm_enroll_iterations,
            random_state=self.rng,  # TODO
            update_means=True,
            update_variances=True,  # TODO default?
            update_weights=True,  # TODO default?
        )
        gmm.variance_thresholds = self.variance_threshold
        gmm = gmm.fit(array)
        return gmm

    def enroll(self, data):
        """Enrolls a GMM using MAP adaptation, given a list of 2D numpy.ndarray's of feature vectors"""
        [self._check_feature(feature) for feature in data]
        array = numpy.vstack(data)
        # Use the array to train a GMM and return it
        return self.enroll_gmm(array)

    ######################################################
    #                Feature comparison                  #
    def read_model(self, model_file):
        """Reads the model, which is a GMM machine"""
        return GMMMachine.from_hdf5(bob.io.base.HDF5File(model_file))

    def score(self, biometric_reference: GMMMachine, data: GMMStats):
        """Computes the score for the given model and the given probe.

        Uses the scoring function passed during initialization.

        Parameters
        ----------
        biometric_reference:
            The model to score against.
        data:
            The probe data to compare to the model.
        """

        assert isinstance(biometric_reference, GMMMachine)  # TODO is it a list?
        assert isinstance(data, GMMStats)
        return self.scoring_function(
            models_means=[biometric_reference],
            ubm=self.ubm,
            test_stats=data,
            frame_length_normalisation=True,
        )[0, 0]

    def score_multiple_biometric_references(
        self, biometric_references: "list[GMMMachine]", data: GMMStats
    ):
        """Computes the score between multiple models and one probe.

        Uses the scoring function passed during initialization.

        Parameters
        ----------
        biometric_references:
            The models to score against.
        data:
            The probe data to compare to the models.
        """

        assert isinstance(biometric_references, GMMMachine)  # TODO is it a list?
        assert isinstance(data, GMMStats)
        return self.scoring_function(
            models_means=biometric_references,
            ubm=self.ubm,
            test_stats=data,
            frame_length_normalisation=True,
        )

    # def score_for_multiple_probes(self, model, probes):
    #     """This function computes the score between the given model and several given probe files."""
    #     assert isinstance(model, GMMMachine)
    #     for probe in probes:
    #         assert isinstance(probe, GMMStats)
    #     #    logger.warn("Please verify that this function is correct")
    #     return self.probe_fusion_function(
    #         self.scoring_function(
    #             model.means, self.ubm, probes, [], frame_length_normalisation=True
    #         )
    #     )

    def fit(self, X, y=None, **kwargs):
        """Trains the UBM."""
        self.train_ubm(X)
        return self

    def transform(self, X, **kwargs):
        """Passthrough. Enroll applies a different transform as score."""
        return X


class GMMRegular(GMM):
    """Algorithm for computing Universal Background Models and Gaussian Mixture Models of the features"""

    def __init__(self, **kwargs):
        """Initializes the local UBM-GMM tool chain with the given file selector object"""
        #    logger.warn("This class must be checked. Please verify that I didn't do any mistake here. I had to rename 'train_projector' into a 'train_enroller'!")
        # initialize the UBMGMM base class
        GMM.__init__(self, **kwargs)
        # register a different set of functions in the Tool base class
        BioAlgorithm.__init__(
            self, requires_enroller_training=True, performs_projection=False
        )

    #######################################################
    #                UBM training                         #

    def train_enroller(self, train_features, enroller_file):
        """Computes the Universal Background Model from the training ("world") data"""
        train_features = [feature for client in train_features for feature in client]
        return self.train_projector(train_features, enroller_file)

    #######################################################
    #              GMM training using UBM                 #

    def load_enroller(self, enroller_file):
        """Reads the UBM model from file"""
        return self.load_projector(enroller_file)

    ######################################################
    #                Feature comparison                  #
    def score(self, model, probe):
        """Computes the score for the given model and the given probe.
        The score are Log-Likelihood.
        Therefore, the log of the likelihood ratio is obtained by computing the following difference."""

        assert isinstance(model, GMMMachine)
        self._check_feature(probe)
        score = sum(
            model.log_likelihood(probe[i, :]) - self.ubm.log_likelihood(probe[i, :])
            for i in range(probe.shape[0])
        )
        return score / probe.shape[0]

    def score_for_multiple_probes(self, model, probes):
        raise NotImplementedError("Implement Me!")
