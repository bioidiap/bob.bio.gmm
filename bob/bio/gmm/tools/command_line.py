import os
import sys
import types

import bob.core
logger = bob.core.log.setup("bob.bio.gmm")

from bob.bio.base.tools import FileSelector

def add_parallel_gmm_options(parsers, sub_module = None):
  """Add the options for parallel UBM training to the given parsers."""

  flag_group = parsers['flag']
  flag_group.add_argument('-l', '--limit-training-data', type=int,
      help = 'Limit the number of training examples used for KMeans initialization and the GMM initialization')

  flag_group.add_argument('-k', '--kmeans-start-iteration', type=int, default=0,
      help = 'Specify the first iteration for the KMeans training (i.e. to restart from there)')
  flag_group.add_argument('-m', '--gmm-start-iteration', type=int, default=0,
      help = 'Specify the first iteration for the GMM training (i.e. to restart from there)')
  flag_group.add_argument('-C', '--clean-intermediate', action='store_true',
      help = 'Clean up temporary files of older iterations?')

  sub_dir_group = parsers['sub-dir']
  sub_dir_group.add_argument('--kmeans-directory', default = 'kmeans_temp',
      help = 'The sub-directory (relative to --temp-directory), where intermediate kmeans files should be stored')
  sub_dir_group.add_argument('--gmm-directory',  default = 'gmm_temp',
      help = 'The sub-directory (relative to --temp-directory), where intermediate gmm files should be stored')

  if sub_module == 'isv':
    sub_dir_group.add_argument('--isv-directory',  default = 'isv_temp',
        help = 'The sub-directory (relative to --temp-directory), where intermediate isv training files should be stored')



# Functions to be added to the FileSelector class, once it is instantiated
def _kmeans_intermediate_file(self, round):
  return os.path.join(self.directories['kmeans'], 'round_%05d' % round, 'kmeans.hdf5')

def _kmeans_stats_file(self, round, start_index, end_index):
  return os.path.join(self.directories['kmeans'], 'round_%05d' % round, 'stats-%05d-%95d.hdf5' % (start_index, end_index))

def _gmm_intermediate_file(self, round):
  return os.path.join(self.directories['gmm'], 'round_%05d' % round, 'gmm.hdf5')

def _gmm_stats_file(self, round, start_index, end_index):
  return os.path.join(self.directories['gmm'], 'round_%05d' % round, 'stats-%05d-%95d.hdf5' % (start_index, end_index))


def initialize_parallel_gmm(args, sub_module = None):
  # get the relevant sub_directory, which depends on the database and the prorocol
  protocol = 'None' if args.database.protocol is None else args.database.protocol
  extractor_sub_dir = protocol if args.database.training_depends_on_protocol and args.extractor.requires_training else '.'
  sub_dir = protocol if args.database.training_depends_on_protocol else '.'

  fs = FileSelector.instance()

  # add relevant **functions** to file selector object
  fs.kmeans_intermediate_file = types.MethodType(_kmeans_intermediate_file, fs)
  fs.kmeans_stats_file =  types.MethodType(_kmeans_stats_file, fs)
  fs.gmm_intermediate_file = types.MethodType(_gmm_intermediate_file, fs)
  fs.gmm_stats_file = types.MethodType(_gmm_stats_file, fs)

  # add relevant directories to file selector object
  fs.directories['kmeans'] = os.path.join(args.temp_directory, sub_dir, args.kmeans_directory)
  fs.kmeans_file = os.path.join(args.temp_directory, sub_dir, "kmeans.hdf5")
  fs.directories['gmm'] = os.path.join(args.temp_directory, sub_dir, args.gmm_directory)
  if sub_module is None:
    fs.ubm_file = fs.projector_file
  else:
    fs.ubm_file = os.path.join(args.temp_directory, sub_dir, "ubm.hdf5")
    fs.directories['isv'] = os.path.join(args.temp_directory, sub_dir, args.isv_directory)
