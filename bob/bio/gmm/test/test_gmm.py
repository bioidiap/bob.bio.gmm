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
import tempfile

import pkg_resources

import bob.bio.gmm

from bob.bio.base.test import utils
from bob.bio.gmm.algorithm import GMM
from bob.learn.em.mixture import GMMMachine
from bob.learn.em.mixture import GMMStats

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


def test_training():
    """Tests the generation of the UBM."""
    # Set a small training iteration count
    gmm1 = GMM(
        number_of_gaussians=2,
        kmeans_training_iterations=1,
        ubm_training_iterations=1,
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
    feature = utils.random_array((20, 45), -5.0, 5.0, seed=84)
    projected = gmm1.project(feature)
    assert isinstance(projected, bob.learn.em.mixture.GMMStats)

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
    enroll = utils.random_training_set((20, 45), 5, -5.0, 5.0, seed=21)
    biometric_reference = gmm1.enroll(enroll)
    assert not biometric_reference.is_similar_to(biometric_reference.ubm)
    assert isinstance(biometric_reference, GMMMachine)

    reference_file = pkg_resources.resource_filename(
        "bob.bio.gmm.test", "data/gmm_enrolled.hdf5"
    )
    if regenerate_refs:
        biometric_reference.save(reference_file)

    gmm2 = gmm1.read_biometric_reference(reference_file)
    assert biometric_reference.is_similar_to(gmm2)


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

    reference_score = 0.045073
    assert (
        abs(gmm1.score(biometric_reference, probe) - reference_score) < 1e-5
    ), "The scores differ: %3.8f, %3.8f" % (
        gmm1.score(biometric_reference, probe),
        reference_score,
    )
    assert (
        abs(
            gmm1.score_for_multiple_probes(biometric_reference, [probe, probe])
            - reference_score
        )
        < 1e-5
    )
