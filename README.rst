.. vim: set fileencoding=utf-8 :
.. Sun Aug 21 21:38:15 CEST 2016

.. image:: http://img.shields.io/badge/docs-v3.2.1-yellow.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.bio.gmm/v3.2.1/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.bio.gmm/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.bio.gmm/badges/v3.2.1/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.gmm/commits/v3.2.1
.. image:: https://gitlab.idiap.ch/bob/bob.bio.gmm/badges/v3.2.1/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.gmm/commits/v3.2.1
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.gmm
.. image:: http://img.shields.io/pypi/v/bob.bio.gmm.svg
   :target: https://pypi.python.org/pypi/bob.bio.gmm


============================================
 Run Gaussian mixture model based algorithms
============================================

This package is part of the signal-processing and machine learning toolbox
Bob_.
This package is part of the ``bob.bio`` packages, which allow to run comparable and reproducible biometric recognition experiments on publicly available databases.

This package contains functionality to run biometric recognition algorithms based on Gaussian mixture models (GMMs).
It is an extension to the `bob.bio.base <http://pypi.python.org/pypi/bob.bio.base>`_ package, which provides the basic scripts.
In this package, utilities that are specific for using GMM-based algorithms are defined:

* Recognition algorithms based on Gaussian mixture models
* Scripts to run the training of these algorithms in parallel



Installation
------------

Complete Bob's `installation`_ instructions. Then, to install this package,
run::

  $ conda install bob.bio.gmm


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss