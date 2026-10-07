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

Reference
---------

To learn more about how dynamite works under the hood, and to see performance data, you may be interested to read `Chapter 2 of Greg's PhD thesis <https://gmeyer.net/dissertation/Ch2.html>`_.

If you use dynamite in a publication, please cite it using the following BibTeX:

.. code-block:: bibtex

   @phdthesis{kahanamoku-meyer_exploring_2023,
       title = {Exploring the {{Limits}} of {{Classical Simulation}}: {{From Computational Many-Body Dynamics}} to {{Quantum Advantage}}},
       author = {{Kahanamoku-Meyer}, Gregory Donald},
       year = 2023,
       school = {University of California, Berkeley},
       isbn = {979-8-3803-6742-4},
       url = {https://escholarship.org/uc/item/6gb6v2j3}
   }

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
