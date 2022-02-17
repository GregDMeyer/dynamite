.. dynamite documentation master file, created by
   sphinx-quickstart on Tue May  2 14:26:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dynamite: fast full quantum dynamics
====================================

Welcome to **dynamite**, which provides a simple interface
to fast evolution of quantum dynamics and eigensolving. Behind the
scenes, dynamite uses the PETSc/SLEPc implementations of Krylov subspace
exponentiation and eigensolving.

For illustrative examples, check out dynamite on
`GitHub! <https://github.com/GregDMeyer/dynamite>`_

The techniques implemented by dynamite fill a niche for numerical
quantum simulation. DMRG methods are very fast but their speed depends on
entanglement, preventing evolution past moderate time scales for systems
that become highly entangled. Exact diagonalization allows for evolution
to arbitrarily long times, but is quite limited in Hilbert space size.
dynamite is best at evolving for moderate time scales (perhaps :math:`\sim 10^4 * 1/|J|`)
on moderate size Hilbert spaces (up to :math:`\sim 30` spins or so).

.. note::
    dynamite is in beta! You may find bugs. When you do,
    please submit them on the `GitHub Issues <https://github.com/GregDMeyer/dynamite/issues>`_
    page! Additionally, you may want to check you are getting correct answers by
    comparing a small system to output from a different method.

Features
--------
 - Easy building of spin chain Hamiltonians through Python
 - Performance-critical code written in C, giving speed comparable
   to pure C implementation
 - Underlying PETSc/SLEPc libraries supply fast algorithms for matrix
   exponentiation and eigensolving
 - Options such as shell matrices provide customizability and access to
   extremely large Hilbert spaces

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   containers.rst
   install.rst
   tips.rst
   FAQ.rst
   dynamite.rst

This package was created by Greg Meyer in `Prof. Norman Yao's lab <https://quantumoptics.physics.berkeley.edu/>`_ at UC Berkeley.
