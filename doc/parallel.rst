
.. _bob.bio.gmm.parallel:

==================================
Executing the Training in Parallel
==================================



Sometimes the training of the GMM-based models require a lot of time.
However, the training procedures can be parallelized, i.e., by running the E-steps of the EM loop in parallel.
For this purpose, we provide a set of scripts ``./bin/verify_gmm.py``, ``./bin/verify_isv.py`` and ``./bin/verify_ivector.py``.
These scripts integrate perfectly into the ``bob.bio`` packages.
Particularly, they have exactly the same set of options as documented in :ref:`bob.bio.base.experiments`.

In fact, the scripts above only run in parallelized mode, i.e., either the ``--grid`` or ``--parallel`` option is required.
During the submission of the jobs, several hundred jobs might be created (depending on the ``number_of_..._training_iterations``  that you specify in the :py:class:`bob.bio.gmm.algorithms.GMM` constructor).
However, after the training has finished, it is possible to use the normal ``./bin/verify.py`` script to run similar experiments, e.g., if you want to change the protocol of your experiment.

.. todo:: improve the documentation of the parallelized scripts.
