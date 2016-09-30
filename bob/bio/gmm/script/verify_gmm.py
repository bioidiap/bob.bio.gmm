#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
from __future__ import print_function

import sys
import argparse

import logging
logger = logging.getLogger("bob.bio.gmm")

import bob.bio.base
from .. import tools, algorithm
from bob.bio.base import tools as base_tools


def parse_arguments(command_line_parameters, exclude_resources_from = []):
  """This function parses the given options (which by default are the command line options). If exclude_resources_from is specified (as a list), the resources from the given packages are not listed in the help message."""
  # set up command line parser
  parsers = base_tools.command_line_parser(exclude_resources_from = exclude_resources_from)

  # add GMM-related options
  tools.add_parallel_gmm_options(parsers)

  # override some parameters
  parsers['config'].add_argument('-a', '--algorithm', metavar = 'x', nargs = '+', default = ['gmm'],
      help = 'Face recognition; only GMM-related algorithms are allowed')


  # Add sub-tasks that can be executed by this script
  parser = parsers['main']
  parser.add_argument('--sub-task',
      choices = ('preprocess', 'train-extractor', 'extract', 'normalize-features', 'kmeans-init', 'kmeans-e-step', 'kmeans-m-step', 'gmm-init', 'gmm-e-step', 'gmm-m-step', 'project', 'enroll', 'compute-scores', 'concatenate'),
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--iteration', type = int,
      help = argparse.SUPPRESS) #'Which type of models to generate (Normal or TModels)'
  parser.add_argument('--model-type', choices = ['N', 'T'],
      help = argparse.SUPPRESS) #'Which type of models to generate (Normal or TModels)'
  parser.add_argument('--score-type', choices = ['A', 'B', 'C', 'D', 'Z'],
      help = argparse.SUPPRESS) #'The type of scores that should be computed'
  parser.add_argument('--group',
      help = argparse.SUPPRESS) #'The group for which the current action should be performed'

  # now that we have set up everything, get the command line arguments
  args = base_tools.initialize(parsers, command_line_parameters,
      skips = ['preprocessing', 'extractor-training', 'extraction', 'normalization', 'kmeans', 'gmm', 'projection', 'enroller-training', 'enrollment', 'score-computation', 'concatenation', 'calibration']
  )

  if args.grid is None and args.parallel is None:
    raise ValueError("To be able to run the parallelized ISV script, either the --grid or the --parallel option need to be specified!")

  args.skip_projector_training = True

  # and add the GMM-related parameters
  tools.initialize_parallel_gmm(args)

  # assert that the algorithm is a GMM
  if tools.base(args.algorithm).__class__ not in (algorithm.GMM, algorithm.GMMRegular):
    raise ValueError("The given algorithm %s is not a (pure) GMM algorithm" % type(args.algorithm))

  # check if one of the parameters is given wothout the sub-task
  if args.sub_task is None:
    if args.iteration is not None: raise ValueError("The option --iteration is an internal option and cannot be used to define experiments")
    if args.model_type is not None: raise ValueError("The option --model-type is an internal option and cannot be used to define experiments")
    if args.score_type is not None: raise ValueError("The option --score-type is an internal option and cannot be used to define experiments")
    if args.group is not None: raise ValueError("The option --group is an internal option and cannot be used to define experiments; did you mean to use --groups?")

  return args

def add_gmm_jobs(args, job_ids, deps, submitter):
  """Adds all GMM-related jobs."""

  algorithm = tools.base(args.algorithm)

  # KMeans
  if not args.skip_kmeans:
    # initialization
    if not args.kmeans_start_iteration:
      job_ids['kmeans-init'] = submitter.submit(
              '--sub-task kmeans-init',
              name = 'k-init',
              dependencies = deps,
              **args.grid.training_queue)
      deps.append(job_ids['kmeans-init'])

    # several iterations of E and M steps
    for iteration in range(args.kmeans_start_iteration, algorithm.kmeans_training_iterations):
      # E-step
      job_ids['kmeans-e-step'] = submitter.submit(
              '--sub-task kmeans-e-step --iteration %d' % iteration,
              name='k-e-%d' % iteration,
              number_of_parallel_jobs = args.grid.number_of_projection_jobs,
              dependencies = [job_ids['kmeans-m-step']] if iteration != args.kmeans_start_iteration else deps,
              **args.grid.projection_queue)

      # M-step
      job_ids['kmeans-m-step'] = submitter.submit(
              '--sub-task kmeans-m-step --iteration %d' % iteration,
              name='k-m-%d' % iteration,
              dependencies = [job_ids['kmeans-e-step']],
              **args.grid.training_queue)

    # add dependence to the last m step
    deps.append(job_ids['kmeans-m-step'])

  # GMM
  if not args.skip_gmm:
    # initialization
    if not args.gmm_start_iteration:
      job_ids['gmm-init'] = submitter.submit(
              '--sub-task gmm-init',
              name = 'g-init',
              dependencies = deps,
              **args.grid.training_queue)
      deps.append(job_ids['gmm-init'])

    # several iterations of E and M steps
    for iteration in range(args.gmm_start_iteration, algorithm.gmm_training_iterations):
      # E-step
      job_ids['gmm-e-step'] = submitter.submit(
              '--sub-task gmm-e-step --iteration %d' % iteration,
              name='g-e-%d' % iteration,
              number_of_parallel_jobs = args.grid.number_of_projection_jobs,
              dependencies = [job_ids['gmm-m-step']] if iteration != args.gmm_start_iteration else deps,
              **args.grid.projection_queue)

      # M-step
      job_ids['gmm-m-step'] = submitter.submit(
              '--sub-task gmm-m-step --iteration %d' % iteration,
              name='g-m-%d' % iteration,
              dependencies = [job_ids['gmm-e-step']],
              **args.grid.training_queue)

    # add dependence to the last m step
    deps.append(job_ids['gmm-m-step'])
  return job_ids, deps




def execute(args):
  """Run the desired job of the tool chain that is specified on command line.
  This job might be executed either in the grid, or locally."""

  # first, let the base script decide if it knows how to execute the job
  if bob.bio.base.script.verify.execute(args):
    return True

  # now, check what we can do
  algorithm = tools.base(args.algorithm)

  # the file selector object
  fs = tools.FileSelector.instance()

  # train the feature projector
  if args.sub_task == 'kmeans-init':
    tools.kmeans_initialize(
        algorithm,
        args.extractor,
        args.limit_training_data,
        force = args.force)

  # train the feature projector
  elif args.sub_task == 'kmeans-e-step':
    tools.kmeans_estep(
        algorithm,
        args.extractor,
        args.iteration,
        indices = base_tools.indices(fs.training_list('extracted', 'train_projector'), args.grid.number_of_projection_jobs),
        force = args.force)

  # train the feature projector
  elif args.sub_task == 'kmeans-m-step':
    tools.kmeans_mstep(
        algorithm,
        args.iteration,
        number_of_parallel_jobs = args.grid.number_of_projection_jobs,
        clean = args.clean_intermediate,
        force = args.force)

  elif args.sub_task == 'gmm-init':
    tools.gmm_initialize(
        algorithm,
        args.extractor,
        args.limit_training_data,
        force = args.force)

  # train the feature projector
  elif args.sub_task == 'gmm-e-step':
    tools.gmm_estep(
        algorithm,
        args.extractor,
        args.iteration,
        indices = base_tools.indices(fs.training_list('extracted', 'train_projector'), args.grid.number_of_projection_jobs),
        force = args.force)

  # train the feature projector
  elif args.sub_task == 'gmm-m-step':
    tools.gmm_mstep(
        algorithm,
        args.iteration,
        number_of_parallel_jobs = args.grid.number_of_projection_jobs,
        clean = args.clean_intermediate,
        force = args.force)
  else:
    # Not our keyword...
    return False
  return True



def verify(args, command_line_parameters, external_fake_job_id = 0):
  """This is the main entry point for computing verification experiments.
  You just have to specify configurations for any of the steps of the toolchain, which are:
  -- the database
  -- the preprocessing
  -- feature extraction
  -- the recognition algorithm
  -- and the grid configuration.
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)."""


  # as the main entry point, check whether the sub-task is specified
  if args.sub_task is not None:
    # execute the desired sub-task
    if not execute(args):
      raise ValueError("The specified --sub-task '%s' is not known to the system" % args.sub_task)
    return {}
  else:
    # add jobs
    submitter = base_tools.GridSubmission(args, command_line_parameters, executable = 'verify_gmm.py', first_fake_job_id = 0)
    retval = tools.add_jobs(args, submitter, local_job_adder = add_gmm_jobs)
    base_tools.write_info(args, command_line_parameters, submitter.executable)

    if args.grid.is_local() and args.run_local_scheduler:
      if args.dry_run:
        print ("Would have started the local scheduler to run the experiments with parallel jobs")
      else:
        # start the jman local deamon
        submitter.execute_local()
      return {}

    else:
      # return job ids as a dictionary
      return retval


def main(command_line_parameters = None):
  """Executes the main function"""
  try:
    # do the command line parsing
    args = parse_arguments(command_line_parameters)

    # perform face verification test
    verify(args, command_line_parameters)
  except Exception as e:
    # track any exceptions as error logs (i.e., to get a time stamp)
    logger.error("During the execution, an exception was raised: %s" % e)
    raise

if __name__ == "__main__":
  main()
