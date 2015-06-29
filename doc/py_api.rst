
==========================
Python API for bob.bio.gmm
==========================

.. todo:: Improve documentation of the functions and classes of bob.bio.gmm.

Generic functions
-----------------

Miscellaneous functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.get_config


Tools to run recognition experiments
------------------------------------

Command line generation
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.gmm.tools.add_parallel_gmm_options
   bob.bio.gmm.tools.initialize_parallel_gmm
   bob.bio.gmm.tools.add_jobs

Parallel GMM
~~~~~~~~~~~~

.. autosummary::
   bob.bio.gmm.tools.kmeans_initialize
   bob.bio.gmm.tools.kmeans_estep
   bob.bio.gmm.tools.kmeans_mstep
   bob.bio.gmm.tools.gmm_initialize
   bob.bio.gmm.tools.gmm_estep
   bob.bio.gmm.tools.gmm_mstep
   bob.bio.gmm.tools.gmm_project

Parallel ISV
~~~~~~~~~~~~

.. autosummary::
   bob.bio.gmm.tools.train_isv

Parallel I-Vector
~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.gmm.tools.ivector_estep
   bob.bio.gmm.tools.ivector_mstep
   bob.bio.gmm.tools.ivector_project
   bob.bio.gmm.tools.train_whitener


Integration with bob.bio.video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.gmm.tools.is_video_extension
   bob.bio.gmm.tools.base
   bob.bio.gmm.tools.read_feature


Details
-------

.. automodule:: bob.bio.gmm.tools


.. include:: links.rst
