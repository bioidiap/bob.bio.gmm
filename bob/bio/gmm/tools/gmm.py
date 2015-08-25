import bob.io.base
import bob.learn.em
import shutil
import numpy
import os

import logging
logger = logging.getLogger("bob.bio.gmm")

from bob.bio.base.tools.FileSelector import FileSelector
from bob.bio.base import utils, tools
from .utils import read_feature


def kmeans_initialize(algorithm, extractor, limit_data = None, force = False):
  """Initializes the K-Means training (non-parallel)."""
  fs = FileSelector.instance()

  output_file = fs.kmeans_intermediate_file(0)

  if utils.check_file(output_file, force, 1000):
    logger.info("UBM training: Skipping KMeans initialization since the file '%s' already exists", output_file)
  else:
    # read data
    logger.info("UBM training: initializing kmeans")
    training_list = utils.selected_elements(fs.training_list('extracted', 'train_projector'), limit_data)
    data = numpy.vstack([read_feature(extractor, feature_file) for feature_file in training_list])

    # Perform KMeans initialization
    kmeans_machine = bob.learn.em.KMeansMachine(algorithm.gaussians, data.shape[1])
    # Creates the KMeansTrainer and call the initialization procedure
    algorithm.kmeans_trainer.initialize(kmeans_machine, data, algorithm.rng)
    bob.io.base.create_directories_safe(os.path.dirname(output_file))
    kmeans_machine.save(bob.io.base.HDF5File(output_file, 'w'))
    logger.info("UBM training: saved initial KMeans machine to '%s'", output_file)


def kmeans_estep(algorithm, extractor, iteration, indices, force=False):
  """Performs a single E-step of the K-Means algorithm (parallel)"""
  if indices[0] >= indices[1]:
    return

  fs = FileSelector.instance()

  # check if we need to compute this step
  stats_file = fs.kmeans_stats_file(iteration, indices[0], indices[1])
  new_machine_file = fs.kmeans_intermediate_file(iteration + 1)

  if  utils.check_file(stats_file, force, 1000) or utils.check_file(new_machine_file, force, 1000):
    logger.info("UBM training: Skipping KMeans E-Step since the file '%s' or '%s' already exists", stats_file, new_machine_file)
  else:
    training_list = fs.training_list('extracted', 'train_projector')
    last_machine_file = fs.kmeans_intermediate_file(iteration)
    kmeans_machine = bob.learn.em.KMeansMachine(bob.io.base.HDF5File(last_machine_file))

    logger.info("UBM training: KMeans E-Step round %d from range(%d, %d)", iteration, *indices)

    # read data
    data = numpy.vstack([read_feature(extractor, training_list[index]) for index in range(indices[0], indices[1])])

    # Performs the E-step
    trainer = algorithm.kmeans_trainer
    trainer.e_step(kmeans_machine, data)

    # write results to file
    dist = numpy.array(trainer.average_min_distance)
    nsamples = numpy.array([indices[1] - indices[0]], dtype=numpy.float64)

    # write statistics
    bob.io.base.create_directories_safe(os.path.dirname(stats_file))
    hdf5 = bob.io.base.HDF5File(stats_file, 'w')
    hdf5.set('zeros', trainer.zeroeth_order_statistics)
    hdf5.set('first', trainer.first_order_statistics)
    hdf5.set('dist', dist * nsamples)
    hdf5.set('nsamples', nsamples)

    logger.info("UBM training: Wrote Stats file '%s'", stats_file)



def _read_stats(filename):
  """Reads accumulated K-Means statistics from file"""
  logger.debug("UBM training: Reading stats file '%s'", filename)
  hdf5 = bob.io.base.HDF5File(filename)
  zeroeth  = hdf5.read('zeros')
  first    = hdf5.read('first')
  nsamples = hdf5.read('nsamples')
  dist     = hdf5.read('dist')
  return (zeroeth, first, nsamples, dist)

def _accumulate(filenames):
  zeroeth, first, nsamples, dist = _read_stats(filenames[0])
  for stat in filenames[1:]:
    zeroeth_, first_, nsamples_, dist_ = _read_stats(stat)
    zeroeth += zeroeth_
    first += first_
    nsamples += nsamples_
    dist += dist_
  return (zeroeth, first, nsamples, dist)

def kmeans_mstep(algorithm, iteration, number_of_parallel_jobs, force=False, clean=False):
  """Performs a single M-step of the K-Means algorithm (non-parallel)"""
  fs = FileSelector.instance()

  old_machine_file = fs.kmeans_intermediate_file(iteration)
  new_machine_file = fs.kmeans_intermediate_file(iteration+1)

  if  utils.check_file(new_machine_file, force, 1000):
    logger.info("UBM training: Skipping KMeans M-Step since the file '%s' already exists", new_machine_file)
  else:
    # get the files from e-step
    training_list = fs.training_list('extracted', 'train_projector')

    # try if there is one file containing all data
    if os.path.exists(fs.kmeans_stats_file(iteration, 0, len(training_list))):
      stats_file = fs.kmeans_stats_file(iteration, 0, len(training_list))
      # load stats file
      statistics = _read_stats(stats_file)
    else:
      # load several files
      filenames = []
      for job in range(number_of_parallel_jobs):
        job_indices = tools.indices(training_list, number_of_parallel_jobs, job+1)
        if job_indices[-1] > job_indices[0]:
          filenames.append(fs.kmeans_stats_file(iteration, job_indices[0], job_indices[-1]))
      statistics = _accumulate(filenames)

    # Creates the KMeansMachine
    kmeans_machine = bob.learn.em.KMeansMachine(bob.io.base.HDF5File(old_machine_file))
    trainer = algorithm.kmeans_trainer
    trainer.reset_accumulators(kmeans_machine)

    trainer.zeroeth_order_statistics = statistics[0]
    trainer.first_order_statistics = statistics[1]
    trainer.average_min_distance = statistics[3]
    error = statistics[3] / statistics[2]

    # Performs the M-step
    trainer.m_step(kmeans_machine, None) # data is not used in M-step
    logger.info("UBM training: Performed M step %d with result %f" % (iteration, error))

    # Save the K-Means model
    bob.io.base.create_directories_safe(os.path.dirname(new_machine_file))
    kmeans_machine.save(bob.io.base.HDF5File(new_machine_file, 'w'))

  # copy the k_means file, when last iteration
  # TODO: implement other stopping criteria
  if iteration == algorithm.kmeans_training_iterations-1:
    shutil.copy(new_machine_file, fs.kmeans_file)
    logger.info("UBM training: Wrote new KMeans machine '%s'", fs.kmeans_file)

  if clean and iteration > 0:
    old_dir = os.path.dirname(fs.kmeans_intermediate_file(iteration-1))
    logger.info("Removing old intermediate directory '%s'", old_dir)
    shutil.rmtree(old_dir)



def gmm_initialize(algorithm, extractor, limit_data = None, force = False):
  """Initializes the GMM calculation with the result of the K-Means algorithm (non-parallel).
  This might require a lot of memory."""
  fs = FileSelector.instance()

  output_file = fs.gmm_intermediate_file(0)

  if utils.check_file(output_file, force, 800):
    logger.info("UBM Training: Skipping GMM initialization since '%s' already exists", output_file)
  else:
    logger.info("UBM Training: Initializing GMM")

    # read features
    training_list = utils.selected_elements(fs.training_list('extracted', 'train_projector'), limit_data)
    data = numpy.vstack([read_feature(extractor, feature_file) for feature_file in training_list])

    # get means and variances of kmeans result
    kmeans_machine = bob.learn.em.KMeansMachine(bob.io.base.HDF5File(fs.kmeans_file))
    [variances, weights] = kmeans_machine.get_variances_and_weights_for_each_cluster(data)

    # Create initial GMM Machine
    gmm_machine = bob.learn.em.GMMMachine(algorithm.gaussians, data.shape[1])

    # Initializes the GMM
    gmm_machine.means = kmeans_machine.means
    gmm_machine.variances = variances
    gmm_machine.weights = weights
    gmm_machine.set_variance_thresholds(algorithm.variance_threshold)

    # write gmm machine to file
    bob.io.base.create_directories_safe(os.path.dirname(output_file))
    gmm_machine.save(bob.io.base.HDF5File(output_file, 'w'))
    logger.info("UBM Training: Wrote GMM file '%s'", output_file)


def gmm_estep(algorithm, extractor, iteration, indices, force=False):
  """Performs a single E-step of the GMM training (parallel)."""
  if indices[0] >= indices[1]:
    return
  fs = FileSelector.instance()

  stats_file = fs.gmm_stats_file(iteration, indices[0], indices[1])
  new_machine_file = fs.gmm_intermediate_file(iteration + 1)

  if  utils.check_file(stats_file, force, 1000) or utils.check_file(new_machine_file, force, 1000):
    logger.info("UBM training: Skipping GMM E-Step since the file '%s' or '%s' already exists", stats_file, new_machine_file)
  else:
    training_list = fs.training_list('extracted', 'train_projector')
    last_machine_file = fs.gmm_intermediate_file(iteration)
    gmm_machine = bob.learn.em.GMMMachine(bob.io.base.HDF5File(last_machine_file))

    logger.info("UBM training: GMM E-Step from range(%d, %d)", *indices)

    # read data
    data = numpy.vstack([read_feature(extractor, training_list[index]) for index in range(indices[0], indices[1])])
    trainer = algorithm.ubm_trainer
    trainer.initialize(gmm_machine, None)

    # Calls the E-step and extracts the GMM statistics
    algorithm.ubm_trainer.e_step(gmm_machine, data)
    gmm_stats = algorithm.ubm_trainer.gmm_statistics

    # Saves the GMM statistics to the file
    bob.io.base.create_directories_safe(os.path.dirname(stats_file))
    gmm_stats.save(bob.io.base.HDF5File(stats_file, 'w'))
    logger.info("UBM training: Wrote GMM stats '%s'", stats_file)


def gmm_mstep(algorithm, iteration, number_of_parallel_jobs, force=False, clean=False):
  """Performs a single M-step of the GMM training (non-parallel)"""
  fs = FileSelector.instance()

  old_machine_file = fs.gmm_intermediate_file(iteration)
  new_machine_file = fs.gmm_intermediate_file(iteration + 1)

  if utils.check_file(new_machine_file, force, 1000):
    logger.info("UBM training: Skipping GMM M-Step since the file '%s' already exists", new_machine_file)
  else:
    # get the files from e-step
    training_list = fs.training_list('extracted', 'train_projector')

    # try if there is one file containing all data
    if os.path.exists(fs.gmm_stats_file(iteration, 0, len(training_list))):
      stats_file = fs.gmm_stats_file(iteration, 0, len(training_list))
      # load stats file
      gmm_stats = bob.learn.em.GMMStats(bob.io.base.HDF5File(stats_file))
    else:
      # load several files
      stats_files = []
      for job in range(number_of_parallel_jobs):
        job_indices = tools.indices(training_list, number_of_parallel_jobs, job+1)
        if job_indices[-1] > job_indices[0]:
          stats_files.append(fs.gmm_stats_file(iteration, job_indices[0], job_indices[-1]))

      # read all stats files
      gmm_stats = bob.learn.em.GMMStats(bob.io.base.HDF5File(stats_files[0]))
      for stats_file in stats_files[1:]:
        gmm_stats += bob.learn.em.GMMStats(bob.io.base.HDF5File(stats_file))

    # load the old gmm machine
    gmm_machine =  bob.learn.em.GMMMachine(bob.io.base.HDF5File(old_machine_file))

    # initialize the trainer
    trainer = algorithm.ubm_trainer
    trainer.initialize(gmm_machine)
    trainer.gmm_statistics = gmm_stats

    # Calls M-step (no data required)
    trainer.m_step(gmm_machine)

    # Saves the GMM statistics to the file
    bob.io.base.create_directories_safe(os.path.dirname(new_machine_file))
    gmm_machine.save(bob.io.base.HDF5File(new_machine_file, 'w'))

  # Write the final UBM file after the last iteration
  # TODO: implement other stopping criteria
  if iteration == algorithm.gmm_training_iterations-1:
    shutil.copy(new_machine_file, fs.ubm_file)
    logger.info("UBM training: Wrote new UBM '%s'", fs.ubm_file)

  if clean and iteration > 0:
    old_dir = os.path.dirname(fs.gmm_intermediate_file(iteration-1))
    logger.info("Removing old intermediate directory '%s'", old_dir)
    shutil.rmtree(old_dir)


def gmm_project(algorithm, extractor, indices, force=False):
  """Performs GMM projection"""
  fs = FileSelector.instance()

  algorithm.load_ubm(fs.ubm_file)

  feature_files = fs.training_list('extracted', 'train_projector')
  projected_files = fs.training_list('projected_gmm', 'train_projector')

  logger.info("ISV training: Project features range (%d, %d) from '%s' to '%s'", indices[0], indices[1], fs.directories['extracted'], fs.directories['projected_gmm'])

  # extract the features
  for i in range(indices[0], indices[1]):
    feature_file = feature_files[i]
    projected_file = projected_files[i]

    if not utils.check_file(projected_file, force):
      # load feature
      feature = read_feature(extractor, feature_file)
      # project feature
      projected = algorithm.project_ubm(feature)
      # write it
      bob.io.base.create_directories_safe(os.path.dirname(projected_file))
      bob.bio.base.save(projected, projected_file)
