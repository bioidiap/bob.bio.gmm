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

import os
import shutil
import numpy
import math
from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_refs = False

seed_value = 5489

import sys
_mac_os = sys.platform == 'darwin'


import scipy.spatial

import bob.io.base
import bob.learn.linear
import bob.io.base.test_utils
import bob.bio.gmm
from bob.bio.base.test import utils

def _compare(data, reference, write_function = bob.bio.base.save, read_function = bob.bio.base.load):
  # write reference?
  if regenerate_refs:
    write_function(data, reference)

  # compare reference
  reference = read_function(reference)
  if hasattr(data, 'is_similar_to'):
    assert data.is_similar_to(reference)
  else:
    assert numpy.allclose(data, reference, atol=1e-5)

def _compare_complex(data, reference, write_function = bob.bio.base.save, read_function = bob.bio.base.load):
  # write reference?
  if regenerate_refs:
    write_function(data, reference)

  # compare reference
  reference = read_function(reference)
  for d,r in zip(data, reference):
    if hasattr(d, 'is_similar_to'):
      assert d.is_similar_to(r)
    else:
      assert numpy.allclose(d, r, atol=1e-5)


def test_gmm():
  temp_file = bob.io.base.test_utils.temporary_filename()
  gmm1 = bob.bio.base.load_resource("gmm", "algorithm")
  assert isinstance(gmm1, bob.bio.gmm.algorithm.GMM)
  assert isinstance(gmm1, bob.bio.base.algorithm.Algorithm)
  assert gmm1.performs_projection
  assert gmm1.requires_projector_training
  assert not gmm1.use_projected_features_for_enrollment
  assert not gmm1.split_training_features_by_client
  assert not gmm1.requires_enroller_training

  # create smaller GMM object
  gmm2 = bob.bio.gmm.algorithm.GMM(
    number_of_gaussians = 2,
    kmeans_training_iterations = 1,
    gmm_training_iterations = 1,
    INIT_SEED = seed_value,
  )

  train_data = utils.random_training_set((100,45), count=5, minimum=-5., maximum=5.)
  reference_file = pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_projector.hdf5')
  try:
    # train the projector
    gmm2.train_projector(train_data, temp_file)

    assert os.path.exists(temp_file)

    if regenerate_refs: shutil.copy(temp_file, reference_file)

    # check projection matrix
    gmm1.load_projector(reference_file)
    gmm2.load_projector(temp_file)

    assert gmm1.ubm.is_similar_to(gmm2.ubm)
  finally:
    if os.path.exists(temp_file): os.remove(temp_file)

  # generate and project random feature
  feature = utils.random_array((20,45), -5., 5., seed=84)
  projected = gmm1.project(feature)
  assert isinstance(projected, bob.learn.em.GMMStats)
  _compare(projected, pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_projected.hdf5'), gmm1.write_feature, gmm1.read_feature)

  # enroll model from random features
  enroll = utils.random_training_set((20,45), 5, -5., 5., seed=21)
  model = gmm1.enroll(enroll)
  assert isinstance(model, bob.learn.em.GMMMachine)
  _compare(model, pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_model.hdf5'), gmm1.write_model, gmm1.read_model)

  # compare model with probe
  probe = gmm1.read_probe(pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_projected.hdf5'))
  reference_score = -0.01676570
  assert abs(gmm1.score(model, probe) - reference_score) < 1e-5, "The scores differ: %3.8f, %3.8f" % (gmm1.score(model, probe), reference_score)
  assert abs(gmm1.score_for_multiple_probes(model, [probe, probe]) - reference_score) < 1e-5


def test_gmm_regular():

  temp_file = bob.io.base.test_utils.temporary_filename()
  gmm1 = bob.bio.base.load_resource("gmm-regular", "algorithm")
  assert isinstance(gmm1, bob.bio.gmm.algorithm.GMMRegular)
  assert isinstance(gmm1, bob.bio.gmm.algorithm.GMM)
  assert isinstance(gmm1, bob.bio.base.algorithm.Algorithm)
  assert not gmm1.performs_projection
  assert not gmm1.requires_projector_training
  assert not gmm1.use_projected_features_for_enrollment
  assert gmm1.requires_enroller_training

  # create smaller GMM object
  gmm2 = bob.bio.gmm.algorithm.GMMRegular(
    number_of_gaussians = 2,
    kmeans_training_iterations = 1,
    gmm_training_iterations = 1,
    INIT_SEED = seed_value,
  )

  train_data = utils.random_training_set((100,45), count=5, minimum=-5., maximum=5.)
  reference_file = pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_projector.hdf5')
  try:
    # train the projector
    gmm2.train_enroller(train_data, temp_file)

    assert os.path.exists(temp_file)

    if regenerate_refs: shutil.copy(temp_file, reference_file)

    # check projection matrix
    gmm1.load_enroller(reference_file)
    gmm2.load_enroller(temp_file)

    assert gmm1.ubm.is_similar_to(gmm2.ubm)
  finally:
    if os.path.exists(temp_file): os.remove(temp_file)

  # enroll model from random features
  enroll = utils.random_training_set((20,45), 5, -5., 5., seed=21)
  model = gmm1.enroll(enroll)
  assert isinstance(model, bob.learn.em.GMMMachine)
  _compare(model, pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_model.hdf5'), gmm1.write_model, gmm1.read_model)

  # generate random probe feature
  probe = utils.random_array((20,45), -5., 5., seed=84)

  # compare model with probe
  reference_score = -0.40840148
  assert abs(gmm1.score(model, probe) - reference_score) < 1e-5, "The scores differ: %3.8f, %3.8f" % (gmm1.score(model, probe), reference_score)
  # TODO: not implemented
  #assert abs(gmm1.score_for_multiple_probes(model, [probe, probe]) - reference_score) < 1e-5


def test_isv():
  temp_file = bob.io.base.test_utils.temporary_filename()
  isv1 = bob.bio.base.load_resource("isv", "algorithm")
  assert isinstance(isv1, bob.bio.gmm.algorithm.ISV)
  assert isinstance(isv1, bob.bio.gmm.algorithm.GMM)
  assert isinstance(isv1, bob.bio.base.algorithm.Algorithm)
  assert isv1.performs_projection
  assert isv1.requires_projector_training
  assert isv1.use_projected_features_for_enrollment
  assert isv1.split_training_features_by_client
  assert not isv1.requires_enroller_training

  # create smaller GMM object
  isv2 = bob.bio.gmm.algorithm.ISV(
      number_of_gaussians = 2,
      subspace_dimension_of_u = 10,
      kmeans_training_iterations = 1,
      gmm_training_iterations = 1,
      isv_training_iterations = 1,
      INIT_SEED = seed_value
  )

  train_data = utils.random_training_set_by_id((100,45), count=5, minimum=-5., maximum=5.)
  reference_file = pkg_resources.resource_filename('bob.bio.gmm.test', 'data/isv_projector.hdf5')
  try:
    # train the projector
    isv2.train_projector(train_data, temp_file)

    assert os.path.exists(temp_file)

    if regenerate_refs: shutil.copy(temp_file, reference_file)

    # check projection matrix
    isv1.load_projector(reference_file)
    isv2.load_projector(temp_file)

    assert isv1.ubm.is_similar_to(isv2.ubm)
    assert isv1.isvbase.is_similar_to(isv2.isvbase)
  finally:
    if os.path.exists(temp_file): os.remove(temp_file)

  # generate and project random feature
  feature = utils.random_array((20,45), -5., 5., seed=84)
  projected = isv1.project(feature)
  assert isinstance(projected, (list, tuple))
  assert len(projected) == 2
  assert isinstance(projected[0], bob.learn.em.GMMStats)
  assert isinstance(projected[1], numpy.ndarray)
  _compare_complex(projected, pkg_resources.resource_filename('bob.bio.gmm.test', 'data/isv_projected.hdf5'), isv1.write_feature, isv1.read_probe)

  # enroll model from random features
  random_features = utils.random_training_set((20,45), count=5, minimum=-5., maximum=5.)
  enroll_features = [isv1.project(feature)[0] for feature in random_features]
  model = isv1.enroll(enroll_features)
  assert isinstance(model, bob.learn.em.ISVMachine)
  _compare(model, pkg_resources.resource_filename('bob.bio.gmm.test', 'data/isv_model.hdf5'), isv1.write_model, isv1.read_model)

  # compare model with probe
  probe = isv1.read_probe(pkg_resources.resource_filename('bob.bio.gmm.test', 'data/isv_projected.hdf5'))
  reference_score = 0.02136784
  assert abs(isv1.score(model, probe) - reference_score) < 1e-5, "The scores differ: %3.8f, %3.8f" % (isv1.score(model, probe), reference_score)
#  assert abs(isv1.score_for_multiple_probes(model, [probe]*4) - reference_score) < 1e-5, isv1.score_for_multiple_probes(model, [probe, probe])
  # TODO: Why is the score not identical for multiple copies of the same probe?
  assert abs(isv1.score_for_multiple_probes(model, [probe, probe]) - reference_score) < 1e-4, isv1.score_for_multiple_probes(model, [probe, probe])


def test_jfa():
  temp_file = bob.io.base.test_utils.temporary_filename()
  jfa1 = bob.bio.base.load_resource("jfa", "algorithm")
  assert isinstance(jfa1, bob.bio.gmm.algorithm.JFA)
  assert isinstance(jfa1, bob.bio.gmm.algorithm.GMM)
  assert isinstance(jfa1, bob.bio.base.algorithm.Algorithm)
  assert jfa1.performs_projection
  assert jfa1.requires_projector_training
  assert jfa1.use_projected_features_for_enrollment
  assert not jfa1.split_training_features_by_client
  assert jfa1.requires_enroller_training

  # create smaller JFA object
  jfa2 = bob.bio.gmm.algorithm.JFA(
      number_of_gaussians = 2,
      subspace_dimension_of_u = 2,
      subspace_dimension_of_v = 2,
      kmeans_training_iterations = 1,
      gmm_training_iterations = 1,
      jfa_training_iterations = 1,
      INIT_SEED = seed_value
  )

  train_data = utils.random_training_set((100,45), count=5, minimum=-5., maximum=5.)
  # reference is the same as for GMM projection
  reference_file = pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_projector.hdf5')
  try:
    # train the projector
    jfa2.train_projector(train_data, temp_file)

    assert os.path.exists(temp_file)

    if regenerate_refs: shutil.copy(temp_file, reference_file)

    # check projection matrix
    jfa1.load_projector(reference_file)
    jfa2.load_projector(temp_file)

    assert jfa1.ubm.is_similar_to(jfa2.ubm)
  finally:
    if os.path.exists(temp_file): os.remove(temp_file)

  # generate and project random feature
  feature = utils.random_array((20,45), -5., 5., seed=84)
  projected = jfa1.project(feature)
  assert isinstance(projected, bob.learn.em.GMMStats)
  _compare(projected, pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_projected.hdf5'), jfa1.write_feature, jfa1.read_feature)

  # enroll model from random features
  random_features = utils.random_training_set_by_id((20,45), count=5, minimum=-5., maximum=5.)
  train_data = [[jfa1.project(feature) for feature in client_features] for client_features in random_features]
  reference_file = pkg_resources.resource_filename('bob.bio.gmm.test', 'data/jfa_enroller.hdf5')
  try:
    # train the projector
    jfa2.train_enroller(train_data, temp_file)

    assert os.path.exists(temp_file)

    if regenerate_refs: shutil.copy(temp_file, reference_file)

    # check projection matrix
    jfa1.load_enroller(reference_file)
    jfa2.load_enroller(temp_file)

    assert jfa1.jfa_base.is_similar_to(jfa2.jfa_base)
  finally:
    if os.path.exists(temp_file): os.remove(temp_file)

  # enroll model from random features
  random_features = utils.random_training_set((20,45), count=5, minimum=-5., maximum=5.)
  enroll_features = [jfa1.project(feature) for feature in random_features]
  model = jfa1.enroll(enroll_features)
  assert isinstance(model, bob.learn.em.JFAMachine)
  _compare(model, pkg_resources.resource_filename('bob.bio.gmm.test', 'data/jfa_model.hdf5'), jfa1.write_model, jfa1.read_model)

  # compare model with probe
  probe = jfa1.read_probe(pkg_resources.resource_filename('bob.bio.gmm.test', 'data/gmm_projected.hdf5'))
  reference_score = 0.02225812
  assert abs(jfa1.score(model, probe) - reference_score) < 1e-5, "The scores differ: %3.8f, %3.8f" % (jfa1.score(model, probe), reference_score)
  # TODO: implement that
  # assert abs(jfa1.score_for_multiple_probes(model, [probe, probe]) - reference_score) < 1e-5, jfa1.score_for_multiple_probes(model, [probe, probe])


"""
  def test10_ivector(self):
    # NOTE: This test will fail when it is run solely. Please always run all Tool tests in order to assure that they work.
    # read input
    feature = facereclib.utils.load(self.input_dir('dct_blocks.hdf5'))
    # assure that the config file is readable
    tool = self.config('ivector')
    self.assertTrue(isinstance(tool, facereclib.tools.IVector))

    # here, we use a reduced complexity for test purposes
    tool = facereclib.tools.IVector(
        number_of_gaussians = 2,
        subspace_dimension_of_t=2,       # T subspace dimension
        update_sigma = False, # TODO Do another test with True
        tv_training_iterations = 1,  # Number of EM iterations for the JFA training
        variance_threshold = 1e-5,
        INIT_SEED = seed_value
    )
    self.assertTrue(tool.performs_projection)
    self.assertTrue(tool.requires_projector_training)
    self.assertTrue(tool.use_projected_features_for_enrollment)
    self.assertFalse(tool.split_training_features_by_client)
    self.assertFalse(tool.requires_enroller_training)

    # train the projector
    t = tempfile.mkstemp('ubm.hdf5', prefix='frltest_')[1]
    tool.train_projector(facereclib.utils.tests.random_training_set(feature.shape, count=5, minimum=-5., maximum=5.), t)
    if regenerate_refs:
      import shutil
      shutil.copy2(t, self.reference_dir('ivector_projector.hdf5'))

    # load the projector file
    tool.load_projector(self.reference_dir('ivector_projector.hdf5'))

    # compare ISV projector with reference
    hdf5file = bob.io.base.HDF5File(t)
    hdf5file.cd('Projector')
    projector_reference = bob.learn.em.GMMMachine(hdf5file)
    self.assertTrue(tool.m_ubm.is_similar_to(projector_reference))

    # compare ISV enroller with reference
    hdf5file.cd('/')
    hdf5file.cd('Enroller')
    enroller_reference = bob.learn.em.IVectorMachine(hdf5file)
    enroller_reference.ubm = projector_reference
    if not _mac_os:
      self.assertTrue(tool.m_tv.is_similar_to(enroller_reference))
    os.remove(t)

    # project the feature
    projected = tool.project(feature)
    if regenerate_refs:
      tool.save_feature(projected, self.reference_dir('ivector_feature.hdf5'))

    # compare the projected feature with the reference
    projected_reference = tool.read_feature(self.reference_dir('ivector_feature.hdf5'))
    self.assertTrue(numpy.allclose(projected,projected_reference))

    # enroll model with the projected feature
    # This is not yet supported
    # model = tool.enroll([projected[0]])
    # if regenerate_refs:
    #  model.save(bob.io.HDF5File(self.reference_dir('ivector_model.hdf5'), 'w'))
    #reference_model = tool.read_model(self.reference_dir('ivector_model.hdf5'))
    # compare the IVector model with the reference
    #self.assertTrue(model.is_similar_to(reference_model))

    # check that the read_probe function reads the correct values
    probe = tool.read_probe(self.reference_dir('ivector_feature.hdf5'))
    self.assertTrue(numpy.allclose(probe,projected))

    # score with projected feature and compare to the weird reference score ...
    # This in not implemented yet

    # score with a concatenation of the probe
    # This is not implemented yet
"""
