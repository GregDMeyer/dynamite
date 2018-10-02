
Tips, Tricks, and Pitfalls
==========================

Pitfalls (TL,DR)
----------------
 - Beware of "nondeterministic" code when running in parallel! Of course your
    code will be technically "deterministic", but I mean code that may run
    differently on different processes. An especially sneaky example is iterating
    through dictionaries: since they are unordered, if you do ``for key,value in d.items():``,
    you may get the items in a different order on different processes. If anything
    in your for loop involves inter-process communication, things will break!
 - It really is useful to read the SLEPc (and PETSc) Users' Manual!


.. _parallelism:

Parallelism
-----------

PETSc and SLEPc are built to leverage massively parallel computing. They use the
MPI (message-passing interface) framework to communicate between processes.
Accordingly, dynamite should be run with MPI. To do so (the ``-n`` flag
specifies the number of processes):

.. code:: bash

    mpirun -n 4 python3 solve_all_the_things.py

Or, if you have MPI installed in an unusual way (e.g. PETSc installed it for
you), you may want:

.. code:: bash

    $PETSC_DIR/bin/petscmpiexec -n 4 python3 solve_quantum_gravity.py

Shell matrices
--------------

Shell, or "matrix-free" matrices, save significantly on memory usage and can
also sometimes speed things up. Instead of storing the entire matrix in memory,
they compute matrix elements on-the-fly when they are needed. When using shell
matrices, the only significant memory usage is the storage of the state vector
(and any other vectors used in the evolution or eigensolve computations).

See ``dynamite.operators.Operator.shell`` for details.

Jupyter Notebook Integration
----------------------------

dynamite integrates well with Jupyter Notebooks, outputting the form of operators
in TeX representation inline. However, getting MPI set up under Jupyter is a bit
of a hassle, so it's best for small computations on one process.

Interacting with PETSc/SLEPc
----------------------------

The underlying ``petsc4py`` matrix for any operator is accessible with
:meth:`dynamite.operators.Operator.get_mat`. For states, the ``petsc4py`` vector
is :attr:`dynamite.states.State.vec`. Arbitrary functions from ``petsc4py`` can
be called through this interface. The documentation is not too extensive for
petsc4py and slepc4py, but it is inferred easily from the C interface.
For example, the C function ``MatMult()`` is implemented as a member function of
the Python :meth:`petsc4py.PETSc.Mat` class: one would just do
``my_matrix.mult(in_vec,result_vec)``.

C programs using PETSc and SLEPc can take options at runtime that modify how the
libraries run. These options can be passed to dynamite as well. It is accomplished by
using :meth:`dynamite.config.initialize`. An example: to change the size of the
Krylov subspace used by SLEPc's matrix exponential, one would do

.. code:: python

    from dynamite import config
    config.initialize(['-mfn_ncv', '40']

GPU Support
-----------

It is possible to run dynamite computations on GPUs, and it is amazingly fast.
PETSc/SLEPc must be built with the ``cuda-opt.py`` configuration script (in
dynamite's ``petsc_config`` directory). Details on using dynamite with GPUs are
coming soon.
