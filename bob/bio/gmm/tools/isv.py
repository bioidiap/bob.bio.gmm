import logging
logger = logging.getLogger("bob.bio.gmm")

import bob.io.base
import os

from bob.bio.base.tools.FileSelector import FileSelector
from bob.bio.base import utils, tools

def train_isv(algorithm, force=False):
  """Finally, the UBM is used to train the ISV projector/enroller."""
  fs = FileSelector.instance()

  if utils.check_file(fs.projector_file, force, 800):
    logger.info("ISV training: Skipping ISV training since '%s' already exists", fs.projector_file)
  else:
    # read UBM into the ISV class
    algorithm.load_ubm(fs.ubm_file)

    # read training data
    training_list = fs.training_list('projected_gmm', 'train_projector', arrange_by_client = True)
    train_gmm_stats = [[algorithm.read_gmm_stats(filename) for filename in client_files] for client_files in training_list]

    # perform ISV training
    logger.info("ISV training: training ISV with %d clients", len(train_gmm_stats))
    algorithm.train_isv(train_gmm_stats)
    # save result
    bob.io.base.create_directories_safe(os.path.dirname(fs.projector_file))
    algorithm.save_projector(fs.projector_file)
