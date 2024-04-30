.. dynamite documentation master file, created by
   sphinx-quickstart on Tue May  2 14:26:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dynamite: fast numerics for quantum many-body spin systems
==========================================================

Welcome to **dynamite**, which provides a simple interface
to fast parallel evolution of quantum dynamics and eigensolving via Krylov subspace methods.

Quick start
-----------

To run the tutorial, `install Docker <containers.html#setup>`_
(or any software that can run docker containers), and run

.. code::

    docker run --rm -p 8887:8887 -w /home/dnm/examples/tutorial gdmeyer/dynamite:latest-jupyter

Then follow the last link in the output (it should start with ``http://127.0.0.1:8887``).
Start the tutorial by launching the notebook ``0-Welcome.ipynb`` in the left panel.

You may also be interested in looking at dynamite's `example scripts <https://github.com/GregDMeyer/dynamite/tree/master/examples/scripts>`_.

.. note::
    dynamite is in beta! You may find bugs. When you do,
    please submit them on the `GitHub Issues <https://github.com/GregDMeyer/dynamite/issues>`_
    page! Additionally, you may want to check you are getting correct answers by
    comparing a small system to output from a different method.

How to cite
-----------

Currently, the best way to cite dynamite is by citing the
`Zenodo repository <https://doi.org/10.5281/zenodo.3606825>`_.
A manuscript is coming soon!

Publications using dynamite
---------------------------

The following list is likely incomplete, please
`let us know <https://github.com/GregDMeyer/dynamite/issues>`_
of any publications that should be added!

.. include:: pubs.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   containers.rst
   install.rst
   FAQ.rst
   dynamite.rst

This package was created by Greg Kahanamoku-Meyer in `Prof. Norman Yao's lab <https://quantumoptics.physics.berkeley.edu/>`_ at UC Berkeley.
