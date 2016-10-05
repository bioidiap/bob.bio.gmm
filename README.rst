.. vim: set fileencoding=utf-8 :
.. Sun Aug 21 21:38:15 CEST 2016

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.bio.gmm/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bob/bob.bio.gmm/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.bio.gmm/badges/v3.0.0/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.gmm/commits/v3.0.0
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.gmm
.. image:: http://img.shields.io/pypi/v/bob.bio.gmm.png
   :target: https://pypi.python.org/pypi/bob.bio.gmm
.. image:: http://img.shields.io/pypi/dm/bob.bio.gmm.png
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

Follow our `installation`_ instructions. Then, using the Python interpreter
provided by the distribution, bootstrap and buildout this package::

  $ python bootstrap-buildout.py
  $ ./bin/buildout


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://gitlab.idiap.ch/bob/bob/wikis/Installation
.. _mailing list: https://groups.google.com/forum/?fromgroups#!forum/bob-devel
