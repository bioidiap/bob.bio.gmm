import logging
logger = logging.getLogger("bob.bio.gmm")

import bob.io.base
import os

from bob.bio.base.tools.FileSelector import FileSelector
from bob.bio.base import utils, tools

def gmm_project(algorithm, extractor, indices, force=False):
  """Performs GMM projection"""
  fs = FileSelector.instance()

  algorithm.load_ubm(fs.ubm_file)

  feature_files = fs.training_list('extracted', 'train_projector')
  projected_files = fs.training_list('isv', 'train_projector')

  logger.info("ISV training: Project features range (%d, %d) from '%s' to '%s'", indices, fs.directories['extracted'], fs.directories['isv'])

  # extract the features
  for i in range(indices[0], indices[1]):
    feature_file = feature_files[i]
    projected_file = projected_files[i]

    if not utils.check_file(projected_file, force):
      # load feature
      feature = extractor.read_feature(feature_file)
      # project feature
      projected = algorithm.project_ubm(feature)
      # write it
      bob.io.base.create_directories_safe(os.path.dirname(projected_file))
      bob.bio.base.save(projected, projected_file)


def isv_training(algorithm, force=False):
  """Finally, the UBM is used to train the ISV projector/enroller."""
  fs = FileSelector.instance()

  if utils.check_file(fs.projector_file, force, 800):
    logger.info("ISV training: Skipping ISV training since '%s' already exists", fs.isv_file)
  else:
    # read UBM into the ISV class
    algorithm.load_ubm(fs.ubm_file)

    # read training data
    training_list = fs.training_list('isv', 'train_projector', arrange_by_client = True)
    train_gmm_stats = [[algorithm.read_gmm_stats(filename) for filename in client_files] for client_files in training_list]

    # perform ISV training
    logger.info("ISV training: training ISV with %d clients", len(train_gmm_stats))
    algorithm.train_isv(train_gmm_stats)
    # save result
    bob.io.base.create_directories_safe(os.path.dirname(fs.projector_file))
    algorithm.save_projector(fs.projector_file)
