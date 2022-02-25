#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Manuel Guenther <Manuel.Guenther@idiap.ch>
# @date: Thu May 24 10:41:42 CEST 2012
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
import tempfile

import numpy
import pkg_resources

import bob.bio.gmm

from bob.bio.base.test import utils
from bob.bio.gmm.algorithm import GMM
from bob.learn.em import GMMMachine
from bob.learn.em import GMMStats

logger = logging.getLogger(__name__)

regenerate_refs = False

seed_value = 5489


def test_class():
    """Tests the creation and initialization of the GMM class."""
    gmm1 = bob.bio.base.load_resource(
        "gmm", "algorithm", preferred_package="bob.bio.gmm"
    )
    assert isinstance(gmm1, GMM)
    assert isinstance(
        gmm1, bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.BioAlgorithm
    )
    assert gmm1.number_of_gaussians == 512
    assert "bob_fit_supports_dask_array" in gmm1._get_tags()
    assert gmm1.transform(None) is None


def test_training():
    """Tests the generation of the UBM."""
    # Set a small training iteration count
    gmm1 = GMM(
        number_of_gaussians=2,
        kmeans_training_iterations=5,
        ubm_training_iterations=5,
        init_seed=seed_value,
    )
    train_data = utils.random_training_set(
        (100, 45), count=5, minimum=-5.0, maximum=5.0
    )

    # Train the UBM (projector)
    gmm1.fit(train_data)

    # Test saving and loading of projector
    with tempfile.NamedTemporaryFile(prefix="bob_", suffix="_model.hdf5") as fd:
        temp_file = fd.name
        gmm1.save_model(temp_file)

        reference_file = pkg_resources.resource_filename(
            "bob.bio.gmm.test", "data/gmm_ubm.hdf5"
        )
        if regenerate_refs:
            gmm1.save_model(reference_file)

        gmm2 = GMM(number_of_gaussians=2)

        gmm2.load_model(temp_file)
        ubm_reference = GMMMachine.from_hdf5(reference_file)
        assert gmm2.ubm.is_similar_to(ubm_reference)


def test_projector():
    """Tests the projector."""
    # Load the UBM
    gmm1 = GMM(number_of_gaussians=2)
    gmm1.ubm = GMMMachine.from_hdf5(
        pkg_resources.resource_filename("bob.bio.gmm.test", "data/gmm_ubm.hdf5")
    )

    # Generate and project random feature
    feature = utils.random_array((20, 45), -5.0, 5.0, seed=seed_value)
    projected = gmm1.project(feature)
    assert isinstance(projected, GMMStats)

    reference_file = pkg_resources.resource_filename(
        "bob.bio.gmm.test", "data/gmm_projected.hdf5"
    )
    if regenerate_refs:
        projected.save(reference_file)

    reference = GMMStats.from_hdf5(reference_file)
    assert projected.is_similar_to(reference)


def test_enroll():
    # Load the UBM
    ubm = GMMMachine.from_hdf5(
        pkg_resources.resource_filename("bob.bio.gmm.test", "data/gmm_ubm.hdf5")
    )
    # Create a GMM object with that UBM
    gmm1 = GMM(
        number_of_gaussians=2, enroll_update_means=True, enroll_update_variances=True
    )
    gmm1.ubm = ubm
    # Enroll the biometric reference from random features
    enroll = utils.random_training_set((20, 45), 5, -5.0, 5.0, seed=seed_value)
    biometric_reference = gmm1.enroll(enroll)
    assert not biometric_reference.is_similar_to(biometric_reference.ubm)
    assert isinstance(biometric_reference, GMMMachine)

    reference_file = pkg_resources.resource_filename(
        "bob.bio.gmm.test", "data/gmm_enrolled.hdf5"
    )
    if regenerate_refs:
        gmm1.write_biometric_reference(biometric_reference, reference_file)

    # Compare to pre-generated file
    gmm2 = gmm1.read_biometric_reference(reference_file)
    assert biometric_reference.is_similar_to(gmm2)

    with tempfile.NamedTemporaryFile(prefix="bob_", suffix="_bioref.hdf5") as fd:
        temp_file = fd.name
        gmm1.write_biometric_reference(biometric_reference, temp_file)
        assert GMMMachine.from_hdf5(temp_file, ubm).is_similar_to(gmm2)


def test_score():
    gmm1 = GMM(number_of_gaussians=2)
    gmm1.load_model(
        pkg_resources.resource_filename("bob.bio.gmm.test", "data/gmm_ubm.hdf5")
    )
    biometric_reference = GMMMachine.from_hdf5(
        pkg_resources.resource_filename("bob.bio.gmm.test", "data/gmm_enrolled.hdf5"),
        ubm=gmm1.ubm,
    )
    probe = GMMStats.from_hdf5(
        pkg_resources.resource_filename("bob.bio.gmm.test", "data/gmm_projected.hdf5")
    )
    probe_data = utils.random_array((20, 45), -5.0, 5.0, seed=seed_value)

    reference_score = 0.601025

    numpy.testing.assert_almost_equal(
        gmm1.score(biometric_reference, probe), reference_score, decimal=5
    )

    multi_probes = gmm1.score_for_multiple_probes(
        biometric_reference, [probe, probe, probe]
    )
    assert multi_probes.shape == (3,), multi_probes.shape
    numpy.testing.assert_almost_equal(multi_probes, reference_score, decimal=5)

    multi_refs = gmm1.score_multiple_biometric_references(
        [biometric_reference, biometric_reference, biometric_reference], probe
    )
    assert multi_refs.shape == (3,), multi_refs.shape
    numpy.testing.assert_almost_equal(multi_refs, reference_score, decimal=5)

    # With not projected data
    numpy.testing.assert_almost_equal(
        gmm1.score(biometric_reference, probe_data), reference_score, decimal=5
    )
