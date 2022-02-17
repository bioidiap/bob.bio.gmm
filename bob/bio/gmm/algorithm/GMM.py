#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Interface between the lower level GMM classes and the Algorithm Transformer.

Implements the enroll and score methods using the low level GMM implementation.

This adds the notions of models, probes, enrollment, and scores to GMM.
"""


import copy
import logging

from typing import Callable

import dask
import dask.array as da
import numpy as np

from h5py import File as HDF5File
from sklearn.base import BaseEstimator

from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import BioAlgorithm
from bob.learn.em.cluster import KMeansMachine
from bob.learn.em.mixture import GMMMachine
from bob.learn.em.mixture import GMMStats
from bob.learn.em.mixture import linear_scoring

logger = logging.getLogger(__name__)

# from bob.pipelines import ToDaskBag  # Used when switching from samples to da.Array


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
        update_means: bool = True,
        update_variances: bool = True,
        update_weights: bool = True,
        # parameters of the GMM enrollment
        gmm_enroll_iterations: int = 1,  # Number of iterations for the enrollment phase
        enroll_update_means: bool = True,
        enroll_update_variances: bool = False,
        enroll_update_weights: bool = False,
        relevance_factor: float = 4,  # Relevance factor as described in Reynolds paper
        responsibility_threshold: float = 0,  # If set, the weight of a particular Gaussian will at least be greater than this threshold. In the case the real weight is lower, the prior mean value will be used to estimate the current mean and variance.
        # scoring
        scoring_function: Callable = linear_scoring,
        # RNG
        init_seed: int = 5489,
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
        gmm_enroll_iterations
            Number of iterations for the MAP GMM used for enrollment.
        enroll_update_weights
            Decides wether the weights of the Gaussians are updated while enrolling.
        enroll_update_means
            Decides wether the means of the Gaussians are updated while enrolling.
        enroll_update_variances
            Decides wether the variancess of the Gaussians are updated while enrolling.
        relevance_factor
            Relevance factor as described in Reynolds paper.
        responsibility_threshold
            If set, the weight of a particular Gaussian will at least be greater than
            this threshold. In the case where the real weight is lower, the prior mean
            value will be used to estimate the current mean and variance.
        init_seed
            Seed for the random number generation.
        scoring_function
            Function returning a score from a model, a UBM, and a probe.
        """

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
        self.enroll_update_means = enroll_update_means
        self.enroll_update_weights = enroll_update_weights
        self.enroll_update_variances = enroll_update_variances
        self.init_seed = init_seed
        self.rng = self.init_seed
        self.responsibility_threshold = responsibility_threshold

        self.scoring_function = scoring_function

        self.ubm = None

        super().__init__()

    def _check_feature(self, feature):
        """Checks that the features are appropriate"""
        if (
            not isinstance(feature, np.ndarray)
            or feature.ndim != 2
            or feature.dtype != np.float64
        ):
            raise ValueError(f"The given feature is not appropriate: \n{feature}")
        if self.ubm is not None and feature.shape[1] != self.ubm.shape[1]:
            raise ValueError(
                "The given feature is expected to have %d elements, but it has %d"
                % (self.ubm.shape[1], feature.shape[1])
            )

    def save_model(self, ubm_file):
        """Saves the projector to file."""
        # Saves the UBM to file
        logger.debug("Saving model to file '%s'", ubm_file)

        hdf5 = ubm_file if isinstance(ubm_file, HDF5File) else HDF5File(ubm_file, "w")
        self.ubm.save(hdf5)

    def load_model(self, ubm_file):
        """Loads the projector from a file."""
        hdf5file = HDF5File(ubm_file, "r")
        logger.debug("Loading model from file '%s'", ubm_file)
        # Read UBM
        self.ubm = GMMMachine.from_hdf5(hdf5file)
        self.ubm.variance_thresholds = self.variance_threshold

    def project(self, array):
        """Computes GMM statistics against a UBM, given a 2D array of feature vectors"""
        self._check_feature(array)
        logger.debug("Projecting %d feature vectors", array.shape[0])
        # Accumulates statistics
        gmm_stats = self.ubm.transform(array)
        gmm_stats.compute()

        # return the resulting statistics
        return gmm_stats

    def read_feature(self, feature_file):
        """Read the type of features that we require, namely GMM_Stats"""
        return GMMStats.from_hdf5(HDF5File(feature_file, "r"))

    def write_feature(self, feature, feature_file):
        """Write the features (GMM_Stats)"""
        return feature.save(feature_file)

    def enroll(self, data):
        """Enrolls a GMM using MAP adaptation given a reference's feature vectors

        Returns a GMMMachine tweaked from the UBM with MAP
        """

        [self._check_feature(feature) for feature in data]
        array = da.vstack(data)
        # Use the array to train a GMM and return it
        logger.debug(" .... Enrolling with %d feature vectors", array.shape[0])

        # TODO responsibility_threshold
        with dask.config.set(scheduler="threads"):
            gmm = GMMMachine(
                n_gaussians=self.number_of_gaussians,
                trainer="map",
                ubm=copy.deepcopy(self.ubm),
                convergence_threshold=self.training_threshold,
                max_fitting_steps=self.gmm_enroll_iterations,
                random_state=self.rng,
                update_means=self.enroll_update_means,
                update_variances=self.enroll_update_variances,
                update_weights=self.enroll_update_weights,
            )
            gmm.variance_thresholds = self.variance_threshold
            gmm.fit(array)
        return gmm

    def read_biometric_reference(self, model_file):
        """Reads an enrolled reference model, which is a MAP GMMMachine"""
        return GMMMachine.from_hdf5(HDF5File(model_file, "r"), ubm=self.ubm)

    @classmethod
    def write_biometric_reference(cls, model: GMMMachine, model_file):
        """Write the enrolled reference (MAP GMMMachine)"""
        return model.save(model_file)

    def score(self, biometric_reference: GMMMachine, probe):
        """Computes the score for the given model and the given probe.

        Uses the scoring function passed during initialization.

        Parameters
        ----------
        biometric_reference:
            The model to score against.
        probe:
            The probe data to compare to the model.
        """

        logger.debug(f"scoring {biometric_reference}, {probe}")
        if not isinstance(probe, GMMStats):
            probe = self.project(
                probe
            )  # Projection is done here instead of transform (or it would be applied to enrollment data too...)
        return self.scoring_function(
            models_means=[biometric_reference],
            ubm=self.ubm,
            test_stats=probe,
            frame_length_normalization=True,
        )[0, 0]

    def score_multiple_biometric_references(
        self, biometric_references: "list[GMMMachine]", probe: GMMStats
    ):
        """Computes the score between multiple models and one probe.

        Uses the scoring function passed during initialization.

        Parameters
        ----------
        biometric_references:
            The models to score against.
        probe:
            The probe data to compare to the models.
        """

        logger.debug(f"scoring {biometric_references}, {probe}")
        assert isinstance(biometric_references[0], GMMMachine), type(
            biometric_references[0]
        )
        stats = self.project(probe) if not isinstance(probe, GMMStats) else probe
        return self.scoring_function(
            models_means=biometric_references,
            ubm=self.ubm,
            test_stats=stats,
            frame_length_normalization=True,
        )

    def score_for_multiple_probes(self, model, probes):
        """This function computes the score between the given model and several given probe files."""
        logger.debug(f"scoring {model}, {probes}")
        assert isinstance(model, GMMMachine)
        stats = [
            self.project(probe) if not isinstance(probe, GMMStats) else probe
            for probe in probes
        ]
        return (
            self.scoring_function(
                models_means=model.means,
                ubm=self.ubm,
                test_stats=stats,
                frame_length_normalization=True,
            )
            .mean()
            .reshape((-1,))
        )

    def fit(self, X, y=None, **kwargs):
        """Trains the UBM."""
        # Stack all the samples in a 2D array of features
        array = da.vstack(X).persist()

        logger.debug("UBM with %d feature vectors", array.shape[0])

        logger.debug(f"Creating UBM machine with {self.number_of_gaussians} gaussians")

        self.ubm = GMMMachine(
            n_gaussians=self.number_of_gaussians,
            trainer="ml",
            max_fitting_steps=self.ubm_training_iterations,
            convergence_threshold=self.training_threshold,
            update_means=self.update_means,
            update_variances=self.update_variances,
            update_weights=self.update_weights,
            k_means_trainer=KMeansMachine(
                self.number_of_gaussians,
                convergence_threshold=self.training_threshold,
                max_iter=self.kmeans_training_iterations,
                init_method="k-means||",
                init_max_iter=5,
            ),
        )

        # Train the GMM
        logger.info("Training UBM GMM")

        self.ubm.fit(array, ubm_train=True)

        return self

    def transform(self, X, **kwargs):
        """Passthrough. Enroll applies a different transform as score."""
        # The idea would be to apply the projection in Transform (going from extracted
        # to GMMStats), but we must not apply this during the training (fit requires
        # extracted data directly).
        # `project` is applied in the score function directly.
        return X

    def _more_tags(self):
        return {"bob_fit_supports_dask_array": True}
