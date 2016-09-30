import bob.measure

import os
import shutil
import tempfile
import numpy
import nose

import bob.io.image
import bob.bio.base
import bob.bio.gmm

import bob.bio.base.test.utils

from nose.plugins.skip import SkipTest

import pkg_resources

regenerate_reference = False

from bob.bio.base.script.verify import main

data_dir = pkg_resources.resource_filename('bob.bio.gmm', 'test/data')

def _verify(parameters, test_dir, sub_dir, ref_modifier="", score_modifier=('scores',''), executable = main):
  try:
    executable(parameters)

    # assert that the score file exists
    score_files = [os.path.join(test_dir, sub_dir, 'Default', norm, '%s-dev%s'%score_modifier) for norm in ('nonorm',  'ztnorm')]
    assert os.path.exists(score_files[0]), "Score file %s does not exist" % score_files[0]
    assert os.path.exists(score_files[1]), "Score file %s does not exist" % score_files[1]

    # also assert that the scores are still the same -- though they have no real meaning
    reference_files = [os.path.join(data_dir, 'scores-%s%s-dev'%(norm, ref_modifier)) for norm in ('nonorm',  'ztnorm')]

    if regenerate_reference:
      for i in (0,1):
        shutil.copy(score_files[i], reference_files[i])

    for i in (0,1):
      d = []
      # read reference and new data
      for score_file in (score_files[i], reference_files[i]):
        f = bob.measure.load.open_file(score_file)
        d_ = []
        for line in f:
          if isinstance(line, bytes): line = line.decode('utf-8')
          d_.append(line.rstrip().split())
        d.append(numpy.array(d_))

      assert d[0].shape == d[1].shape
      # assert that the data order is still correct
      assert (d[0][:,0:3] == d[1][:, 0:3]).all()
      # assert that the values are OK
      assert numpy.allclose(d[0][:,3].astype(float), d[1][:,3].astype(float), 1e-5)

  finally:
    shutil.rmtree(test_dir)


def test_gmm_sequential():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-a', 'bob.bio.gmm.algorithm.GMM(2, 2, 2)',
      '--zt-norm',
      '-vs', 'test_gmm_sequential',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.gmm'
  ]

  _verify(parameters, test_dir, 'test_gmm_sequential', ref_modifier='-gmm')


@bob.bio.base.test.utils.grid_available
def test_gmm_parallel():
  from bob.bio.gmm.script.verify_gmm import main
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  test_database = os.path.join(test_dir, "submitted.sql3")
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-a', 'bob.bio.gmm.algorithm.GMM(2, 2, 2)', '--import', 'bob.bio.gmm', 'bob.io.image',
      '-g', 'bob.bio.base.grid.Grid(grid_type = "local", number_of_parallel_processes = 2, scheduler_sleep_time = 0.1)', '-G', test_database, '--run-local-scheduler', '--stop-on-failure',
      '--clean-intermediate',
      '--zt-norm',
      '-vs', 'test_gmm_parallel',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.gmm'
  ]

  _verify(parameters, test_dir, 'test_gmm_parallel', executable=main, ref_modifier='-gmm')


def test_isv_sequential():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-a', 'bob.bio.gmm.algorithm.ISV(10, number_of_gaussians=2, kmeans_training_iterations=2, gmm_training_iterations=2, isv_training_iterations=2)',
      '--zt-norm',
      '-vs', 'test_isv_sequential',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.gmm'
  ]

  _verify(parameters, test_dir, 'test_isv_sequential', ref_modifier='-isv')


@bob.bio.base.test.utils.grid_available
def test_isv_parallel():
  from bob.bio.gmm.script.verify_isv import main
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  test_database = os.path.join(test_dir, "submitted.sql3")
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-a', 'bob.bio.gmm.algorithm.ISV(10, number_of_gaussians=2, kmeans_training_iterations=2, gmm_training_iterations=2, isv_training_iterations=2)', '--import', 'bob.bio.gmm', 'bob.io.image',
      '-g', 'bob.bio.base.grid.Grid(grid_type = "local", number_of_parallel_processes = 2, scheduler_sleep_time = 0.1)', '-G', test_database, '--run-local-scheduler', '--stop-on-failure',
      '--clean-intermediate',
      '--zt-norm',
      '-vs', 'test_isv_parallel',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.gmm'
  ]

  _verify(parameters, test_dir, 'test_isv_parallel', executable=main, ref_modifier='-isv')


def test_ivector_cosine_sequential():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-a', 'bob.bio.gmm.algorithm.IVector(10, number_of_gaussians=2, kmeans_training_iterations=2, gmm_training_iterations=2, tv_training_iterations=2)',
      '--zt-norm',
      '-vs', 'test_ivector_cosine_sequential',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.gmm'
  ]

  _verify(parameters, test_dir, 'test_ivector_cosine_sequential', ref_modifier='-ivector-cosine')


@bob.bio.base.test.utils.grid_available
def test_ivector_cosine_parallel():
  from bob.bio.gmm.script.verify_ivector import main
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  test_database = os.path.join(test_dir, "submitted.sql3")
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-a', 'bob.bio.gmm.algorithm.IVector(10, number_of_gaussians=2, kmeans_training_iterations=2, gmm_training_iterations=2, tv_training_iterations=2)', '--import', 'bob.bio.gmm', 'bob.io.image',
      '-g', 'bob.bio.base.grid.Grid(grid_type = "local", number_of_parallel_processes = 2, scheduler_sleep_time = 0.1)', '-G', test_database, '--run-local-scheduler', '--stop-on-failure',
      '--clean-intermediate',
      '--zt-norm',
      '-vs', 'test_ivector_cosine_parallel',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.gmm'
  ]

  _verify(parameters, test_dir, 'test_ivector_cosine_parallel', executable=main, ref_modifier='-ivector-cosine')

def test_ivector_lda_wccn_plda_sequential():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-a', 'bob.bio.gmm.algorithm.IVector(10, number_of_gaussians=2, kmeans_training_iterations=2, gmm_training_iterations=2, tv_training_iterations=2, use_lda=True, use_wccn=True, use_plda=True, lda_dim=2, plda_dim_F=2, plda_dim_G=2, plda_training_iterations=2)',
      '--zt-norm',
      '-vs', 'test_ivector_lda_wccn_plda_sequential',
      '--temp-directory', test_dir,
      '--result-directory', test_dir
  ]

  _verify(parameters, test_dir, 'test_ivector_lda_wccn_plda_sequential', ref_modifier='-ivector-lda-wccn-plda')


@bob.bio.base.test.utils.grid_available
def test_ivector_lda_wccn_plda_parallel():
  from bob.bio.gmm.script.verify_ivector import main
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  test_database = os.path.join(test_dir, "submitted.sql3")
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-a', 'bob.bio.gmm.algorithm.IVector(10, number_of_gaussians=2, kmeans_training_iterations=2, gmm_training_iterations=2, tv_training_iterations=2, use_lda=True, use_wccn=True, use_plda=True, lda_dim=2, plda_dim_F=2, plda_dim_G=2, plda_training_iterations=2)', '--import', 'bob.bio.gmm', 'bob.io.image',
      '-g', 'bob.bio.base.grid.Grid(grid_type = "local", number_of_parallel_processes = 2, scheduler_sleep_time = 0.1)', '-G', test_database, '--run-local-scheduler', '--stop-on-failure',
      '--clean-intermediate',
      '--zt-norm',
      '-vs', 'test_ivector_lda_wccn_plda_parallel',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.gmm'
  ]

  _verify(parameters, test_dir, 'test_ivector_lda_wccn_plda_parallel', executable=main, ref_modifier='-ivector-lda-wccn-plda')


def test_internal_raises():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  test_database = os.path.join(test_dir, "submitted.sql3")
  # define dummy parameters
  parameters = [
      '-d', 'dummy',
      '-p', 'dummy',
      '-e', 'dummy2d',
      '-g', 'bob.bio.base.grid.Grid(grid_type = "local", number_of_parallel_processes = 2, scheduler_sleep_time = 0.1)', '-G', test_database, '--run-local-scheduler', '--stop-on-failure',
      '--import', 'bob.bio.gmm', 'bob.io.image',
      '--clean-intermediate',
      '--zt-norm',
      '-vs', 'test_raises',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--preferred-package', 'bob.bio.gmm'
  ]

  from bob.bio.gmm.script.verify_gmm import main as gmm
  from bob.bio.gmm.script.verify_isv import main as isv
  from bob.bio.gmm.script.verify_ivector import main as ivector

  for script, algorithm in (
      (gmm, "bob.bio.gmm.algorithm.GMM(2, 2, 2)"),
      (isv, "bob.bio.gmm.algorithm.ISV(10, number_of_gaussians=2, kmeans_training_iterations=2, gmm_training_iterations=2, isv_training_iterations=2)"),
      (ivector, "bob.bio.gmm.algorithm.IVector(10, number_of_gaussians=2, kmeans_training_iterations=2, gmm_training_iterations=2, tv_training_iterations=2, use_lda=True, use_wccn=True, use_plda=True, lda_dim=2, plda_dim_F=2, plda_dim_G=2, plda_training_iterations=2)")):

    for option, value in (("--iteration", "0"), ("--group", "dev"), ("--model-type", "N"), ("--score-type", "A")):
      internal = parameters + ["--algorithm", algorithm, option, value]

      nose.tools.assert_raises(ValueError, script, internal)
  shutil.rmtree(test_dir)
