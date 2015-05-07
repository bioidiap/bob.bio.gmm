Example buildout environment
============================

This simple example demonstrates how to wrap Bob-based scripts on buildout
environments. This may be useful for homework assignments, tests or as a way to
distribute code to reproduce your publication. In summary, if you need to give
out code to others, we recommend you do it following this template so your code
can be tested, documented and run in an orderly fashion.

Installation
------------

.. note::

  To follow these instructions locally you will need a local copy of this
  package. For that, you can use the github tarball API to download the package::

    $ wget --no-check-certificate https://github.com/idiap/bob.project.example/tarball/master -O- | tar xz
    $ mv idiap-bob.project* bob.project.example

Documentation and Further Information
-------------------------------------

Please refer to the latest Bob user guide, accessing from the `Bob website
<http://idiap.github.com/bob/>`_ for how to create your own packages based on
this example. In particular, the Section entitled `Organize Your Work in
Satellite Packages <http://www.idiap.ch/software/bob/docs/releases/last/sphinx/html/OrganizeYourCode.html>`_
contains details on how to setup, build and roll out your code.
