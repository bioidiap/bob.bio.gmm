#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
from __future__ import print_function

import sys
import argparse

import logging
logger = logging.getLogger("bob.bio.gmm")
from . import verify_isv




def main(command_line_parameters = None):
  """Executes the main function"""
  try:
    # do the command line parsing
    args = verify_isv.parse_arguments(command_line_parameters)
    
    
    args.groups = ['world']
    args.group = 'world'
    args.skip_projection = True
    args.skip_enrollment = True
    args.skip_score_computation = True
    args.skip_concatenation = True
    args.skip_calibration = True

    # perform face verification test
    verify_isv.verify(args, command_line_parameters)
  except Exception as e:
    # track any exceptions as error logs (i.e., to get a time stamp)
    logger.error("During the execution, an exception was raised: %s" % e)
    raise

if __name__ == "__main__":
  main()
