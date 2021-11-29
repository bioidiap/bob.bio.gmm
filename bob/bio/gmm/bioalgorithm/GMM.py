#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

"""Interface between the lower level GMM classes and the Algorithm Transformer.

Implements the enroll and score methods using the low level GMM implementation.

This adds the notions of models, probes, enrollment, and scores to GMM.
"""


import logging

from typing import Callable

import dask.array as da
import numpy as np
import dask
from h5py import File as HDF5File

from sklearn.base import BaseEstimator

import bob.core

from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import BioAlgorithm
from bob.learn.em.mixture import GMMMachine
from bob.learn.em.mixture import GMMStats
from bob.learn.em.mixture import linear_scoring
from bob.pipelines.wrappers import DaskWrapper

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
        self.rng = self.init_seed  # TODO verify if rng object needed
        self.responsibility_threshold = responsibility_threshold

        def scoring_function_wrapped(*args, **kwargs):
            with dask.config.set(scheduler="threads"):
                return scoring_function(*args, **kwargs).compute()

        self.scoring_function = scoring_function_wrapped

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

    def save_ubm(self, ubm_file):
        """Saves the projector to file"""
        # Saves the UBM to file
        logger.debug("Saving model to file '%s'", ubm_file)

        hdf5 = (
            ubm_file
            if isinstance(ubm_file, HDF5File)
            else HDF5File(ubm_file, "w")
        )
        self.ubm.save(hdf5)

    def load_ubm(self, ubm_file):
        hdf5file = HDF5File(ubm_file)
        logger.debug("Loading model from file '%s'", ubm_file)
        # read UBM
        self.ubm = GMMMachine.from_hdf5(hdf5file)
        self.ubm.variance_thresholds = self.variance_threshold

    def project(self, array):
        """Computes GMM statistics against a UBM, given a 2D array of feature vectors"""
        self._check_feature(array)
        logger.debug(" .... Projecting %d feature vectors", array.shape[0])
        # Accumulates statistics
        with dask.config.set(scheduler="threads"):
            gmm_stats = GMMStats(self.ubm.shape[0], self.ubm.shape[1])
            self.ubm.acc_statistics(array, gmm_stats)
            gmm_stats.compute()

        # return the resulting statistics
        return gmm_stats

    def read_feature(self, feature_file):
        """Read the type of features that we require, namely GMM_Stats"""
        return GMMStats.from_hdf5(HDF5File(feature_file))

    def write_feature(self, feature, feature_file):
        """Write the features (GMM_Stats)"""
        return feature.save(feature_file)

    def enroll(self, data):
        """Enrolls a GMM using MAP adaptation, given a list of 2D np.ndarray's of feature vectors"""
        [self._check_feature(feature) for feature in data]
        array = np.vstack(data)
        # Use the array to train a GMM and return it
        logger.debug(" .... Enrolling with %d feature vectors", array.shape[0])

        # TODO responsibility_threshold
        with dask.config.set(scheduler="threads"):
            gmm = GMMMachine(
                n_gaussians=self.number_of_gaussians,
                trainer="map",
                ubm=self.ubm,
                convergence_threshold=self.training_threshold,
                max_fitting_steps=self.gmm_enroll_iterations,
                random_state=self.rng,
                update_means=True,
                update_variances=True,  # TODO default?
                update_weights=True,  # TODO default?
            )
            gmm.variance_thresholds = self.variance_threshold
            gmm = gmm.fit(array)
            # info = {k: type(v) for k, v in gmm.__dict__.items()}
            # for k, v in gmm.gaussians_.__dict__.items():
            #     info[k] = type(v)
            # raise ValueError(str(info))
        return gmm

    def read_model(self, model_file):
        """Reads the model, which is a GMM machine"""
        return GMMMachine.from_hdf5(HDF5File(model_file), ubm=self.ubm)

    def write_model(self, model, model_file):
        """Write the features (GMM_Stats)"""
        return model.save(model_file)

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

        assert isinstance(biometric_reference, GMMMachine)
        stats = self.project(data)
        return self.scoring_function(
            models_means=[biometric_reference],
            ubm=self.ubm,
            test_stats=stats,
            frame_length_normalization=True,
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

        assert isinstance(biometric_references[0], GMMMachine), type(
            biometric_references[0]
        )
        stats = self.project(data)
        return self.scoring_function(
            models_means=biometric_references,
            ubm=self.ubm,
            test_stats=stats,
            frame_length_normalization=True,
        )

    def score_for_multiple_probes(self, model, probes):
        """This function computes the score between the given model and several given probe files."""
        assert isinstance(model, GMMMachine)
        for probe in probes:
            assert isinstance(probe, GMMStats)
        #    logger.warn("Please verify that this function is correct")
        return (
            self.scoring_function(
                models_means=model.means,
                ubm=self.ubm,
                test_stats=probes,
                frame_length_normalization=True,
            )
            .mean()
            .reshape((-1,))
        )

    def fit(self, X, y=None, **kwargs):
        """Trains the UBM."""

        # Stack all the samples in a 2D array of features
        array = da.vstack(X)

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
            # TODO more params?
        )

        # Trains the GMM
        logger.info("Training UBM GMM")
        # Resetting the pseudo random number generator so we can have the same initialization for serial and parallel execution.
        # self.rng = bob.core.random.mt19937(self.init_seed)
        self.ubm.fit(array)

        return self

    def transform(self, X, **kwargs):
        """Passthrough. Enroll applies a different transform as score."""
        # The idea would be to apply the projection in Transform (going from extracted
        # to GMMStats), but we must not apply this during the training (fit requires
        # extracted data directly).
        # `project` is applied in the score function directly.
        return X




def delayed_to_da(delayed, meta=None):
    """Converts one dask.delayed object to a dask.array"""
    if meta is None:
        meta = np.array(delayed.data.compute())

    darray = da.from_delayed(delayed.data, meta.shape, dtype=meta.dtype, name=False)
    return darray, meta


def delayed_samples_to_dask_arrays(delayed_samples, meta=None):
    output = []
    for ds in delayed_samples:
        d_array, meta = delayed_to_da(ds, meta)
        output.append(d_array)
    return output, meta


def delayeds_to_dask_array(delayeds, meta=None):
    """Converts a set of dask.delayed to a list of dask.array"""
    output = []
    for d in delayeds:
        d_array, meta = delayed_samples_to_dask_arrays(d, meta)
        output.extend(d_array)
    return output


class GMMDaskWrapper(DaskWrapper):
    def fit(self, X, y=None, **fit_params):
        # convert X which is a dask bag to a dask array
        X = X.persist()
        delayeds = X.to_delayed()
        lengths = X.map_partitions(lambda samples: [len(samples)]).compute()
        shapes = X.map_partitions(
            lambda samples: [[s.data.shape for s in samples]]
        ).compute()
        dtype, X = None, []
        for l, s, d in zip(lengths, shapes, delayeds):
            d._length = l
            for shape, ds in zip(s, d):
                if dtype is None:
                    dtype = np.array(ds.data.compute()).dtype
                darray = da.from_delayed(ds.data, shape, dtype=dtype, name=False)
                X.append(darray)
        self.estimator.fit(X, y, **fit_params)
        return self
