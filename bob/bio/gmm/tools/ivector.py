import logging
logger = logging.getLogger("bob.bio.gmm")

import bob.io.base
import os
import shutil

from bob.bio.base.tools.FileSelector import FileSelector
from bob.bio.base import utils, tools



def ivector_estep(algorithm, iteration, indices, force=False):
  """Performs a single E-step of the IVector algorithm (parallel)"""
  fs = FileSelector.instance()
  stats_file = fs.ivector_stats_file(iteration, indices[0], indices[1])

  if utils.check_file(stats_file, force, 1000):
    logger.info("IVector training: Skipping IVector E-Step since the file '%s' already exists", stats_file)
  else:
    logger.info("IVector training: E-Step from range(%d, %d)", *indices)

    # Temporary machine used for initialization
    algorithm.load_ubm(fs.ubm_file)

    # get the IVectorTrainer and call the initialization procedure
    trainer = algorithm.ivector_trainer

    # Load machine
    if iteration:
      # load last TV file
      tv = bob.learn.em.IVectorMachine(bob.io.base.HDF5File(fs.ivector_intermediate_file(iteration)))
      tv.ubm = algorithm.ubm
    else:
      # create new TV machine
      tv = bob.learn.em.IVectorMachine(algorithm.ubm, algorithm.subspace_dimension_of_t, algorithm.variance_threshold)
      trainer.initialize(tv)

    # Load data
    training_list = fs.training_list('projected_gmm', 'train_projector')
    data = [algorithm.read_gmm_stats(training_list[i]) for i in range(indices[0], indices[1])]

    # Perform the E-step
    trainer.e_step(tv, data)

    # write results to file
    bob.io.base.create_directories_safe(os.path.dirname(stats_file))
    hdf5 = bob.io.base.HDF5File(stats_file, 'w')
    hdf5.set('acc_nij_wij2', trainer.acc_nij_wij2)
    hdf5.set('acc_fnormij_wij', trainer.acc_fnormij_wij)
    hdf5.set('acc_nij', trainer.acc_nij)
    hdf5.set('acc_snormij', trainer.acc_snormij)
    hdf5.set('nsamples', indices[1] - indices[0])
    logger.info("IVector training: Wrote Stats file '%s'", stats_file)


def _read_stats(filename):
  """Reads accumulated IVector statistics from file"""
  logger.debug("IVector training: Reading stats file '%s'", filename)
  hdf5 = bob.io.base.HDF5File(filename)
  acc_nij_wij2    = hdf5.read('acc_nij_wij2')
  acc_fnormij_wij = hdf5.read('acc_fnormij_wij')
  acc_nij         = hdf5.read('acc_nij')
  acc_snormij     = hdf5.read('acc_snormij')
  return acc_nij_wij2, acc_fnormij_wij, acc_nij, acc_snormij

def _accumulate(filenames):
  acc_nij_wij2, acc_fnormij_wij, acc_nij, acc_snormij = _read_stats(filenames[0])
  for filename in filenames[1:]:
    acc_nij_wij2_, acc_fnormij_wij_, acc_nij_, acc_snormij_ = _read_stats(filename)
    acc_nij_wij2    += acc_nij_wij2_
    acc_fnormij_wij += acc_fnormij_wij_
    acc_nij         += acc_nij_
    acc_snormij     += acc_snormij_
  return acc_nij_wij2, acc_fnormij_wij, acc_nij, acc_snormij


def ivector_mstep(algorithm, iteration, number_of_parallel_jobs, force=False, clean=False):
  """Performs a single M-step of the IVector algorithm (non-parallel)"""
  fs = FileSelector.instance()

  old_machine_file = fs.ivector_intermediate_file(iteration)
  new_machine_file = fs.ivector_intermediate_file(iteration + 1)

  if  utils.check_file(new_machine_file, force, 1000):
    logger.info("IVector training: Skipping IVector M-Step since the file '%s' already exists", new_machine_file)
  else:
    # get the files from e-step
    training_list = fs.training_list('projected_gmm', 'train_projector')
    # try if there is one file containing all data
    if os.path.exists(fs.ivector_stats_file(iteration, 0, len(training_list))):
      # load stats file
      statistics = self._read_stats(fs.ivector_stats_file(iteration, 0, len(training_list)))
    else:
      # load several files
      stats_files = []
      for job in range(number_of_parallel_jobs):
        job_indices = tools.indices(training_list, number_of_parallel_jobs, job+1)
        if job_indices[-1] >= job_indices[0]:
          stats_files.append(fs.ivector_stats_file(iteration, job_indices[0], job_indices[-1]))
      # read all stats files
      statistics = _accumulate(stats_files)

    # Load machine
    algorithm.load_ubm(fs.ubm_file)
    if iteration:
      tv = bob.learn.em.IVectorMachine(bob.io.base.HDF5File(old_machine_file))
      tv.ubm = algorithm.ubm
    else:
      tv = bob.learn.em.IVectorMachine(algorithm.ubm, algorithm.subspace_dimension_of_t, algorithm.variance_threshold)

    # Creates the IVectorTrainer and initialize values
    trainer = algorithm.ivector_trainer
    trainer.reset_accumulators(tv)
    trainer.acc_nij_wij2 = statistics[0]
    trainer.acc_fnormij_wij = statistics[1]
    trainer.acc_nij = statistics[2]
    trainer.acc_snormij = statistics[3]
    trainer.m_step(tv) # data is not used in M-step
    logger.info("IVector training: Performed M step %d", iteration)

    # Save the IVector model
    bob.io.base.create_directories_safe(os.path.dirname(new_machine_file))
    tv.save(bob.io.base.HDF5File(new_machine_file, 'w'))
    logger.info("IVector training: Wrote new IVector machine '%s'", new_machine_file)

  if iteration == algorithm.tv_training_iterations-1:
    shutil.copy(new_machine_file, fs.tv_file)
    logger.info("IVector training: Wrote new TV matrix '%s'", fs.tv_file)

  if clean and iteration > 0:
    old_dir = os.path.dirname(fs.ivector_intermediate_file(iteration-1))
    logger.info("Removing old intermediate directory '%s'", old_dir)
    shutil.rmtree(old_dir)


def ivector_project(algorithm, indices, force=False):
  """Performs IVector projection"""
  # read UBM and TV into the IVector class
  fs = FileSelector.instance()
  algorithm.load_ubm(fs.ubm_file)
  algorithm.load_tv(fs.tv_file)

  gmm_stats_files = fs.training_list('projected_gmm', 'train_projector')
  ivector_files = fs.training_list('projected_ivector', 'train_projector')

  logger.info("IVector training: Project features range (%d, %d) from '%s' to '%s'", indices[0], indices[1], fs.directories['projected_gmm'], fs.directories['projected_ivector'])
  # extract the features
  for i in range(indices[0], indices[1]):
    gmm_stats_file = gmm_stats_files[i]
    ivector_file = ivector_files[i]
    if not utils.check_file(ivector_file, force):
      # load feature
      feature = algorithm.read_gmm_stats(gmm_stats_file)
      # project feature
      projected = algorithm.project_ivector(feature)
      # write it
      bob.io.base.create_directories_safe(os.path.dirname(ivector_file))
      bob.bio.base.save(projected, ivector_file)


def train_whitener(algorithm, force=False):
  """Train the feature projector with the extracted features of the world group."""
  fs = FileSelector.instance()

  if utils.check_file(fs.whitener_file, force, 1000):
    logger.info("- Whitening projector '%s' already exists.", fs.whitener_file)
  else:
    train_files = fs.training_list('projected_ivector', 'train_projector')
    train_features = [bob.bio.base.load(f) for f in train_files]
    # perform training
    algorithm.train_whitener(train_features)
    bob.io.base.create_directories_safe(os.path.dirname(fs.whitener_file))
    bob.bio.base.save(algorithm.whitener, fs.whitener_file)

  # finally, save the projector into one file
  algorithm.load_ubm(fs.ubm_file)
  algorithm.load_tv(fs.tv_file)
  algorithm.load_whitener(fs.whitener_file)
  logger.info("Writing projector into file %s", fs.projector_file)
  algorithm.save_projector(fs.projector_file)


"""
def whitening_ivector(self, indices=None, force=False):
  ""Performs IVector projection""
  # read UBM into the IVector class
  projector = self.m_tool.load_whitening(self.m_configuration.whitening_file)

  projected_ivec_files = self.projected_list(groups=self.groups(), directory=self.m_configuration.ivec_directory)
  projected_whitened_files = self.projected_list(groups=self.groups(), directory=self.m_configuration.whitening_directory)

  # select a subset of indices to iterate
  if indices != None:
    index_range = range(indices[0], indices[1])
    facereclib.utils.info("- Projection: splitting of index range %s" % str(indices))
  else:
    index_range = range(len(projected_ivec_files))

  facereclib.utils.info("- Projection: projecting %d gmm stats from directory '%s' to directory '%s'" % (len(index_range), self.m_configuration.ivec_directory, self.m_configuration.whitening_directory))
  # extract the features
  for i in index_range:
    if not self.m_tool_chain.__check_file__(projected_whitened_files[i], force):
      # load feature
      feature = facereclib.utils.load(str(projected_ivec_files[i]))
      # project feature
      whitened = self.m_tool.project_whitening(feature)
      # write it
      facereclib.utils.ensure_dir(os.path.dirname(projected_whitened_files[i]))
      facereclib.utils.save(whitened, str(projected_whitened_files[i]))
"""
