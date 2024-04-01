
FAQ
===

- :ref:`parallel`
- :ref:`ranks`
- :ref:`shell`
- :ref:`integer`
- :ref:`L`
- :ref:`nondeterm`
- :ref:`petsc`
- :ref:`gpu-aware-mpi`

.. _parallel:

How do I run my code in parallel?
---------------------------------

One of dynamite's most important features is its ability to scale across multiple processors using MPI (or even multiple GPUs!). For example, running with four MPI ranks (processes) is as easy as the following:

.. code:: bash

    mpirun -n 4 python3 solve_all_the_things.py

Note that using MPI with containers can require special steps; see :ref:`containers` for details.

To accelerate dynamite's computations using a GPU, the simplest way is to use one of the GPU-accelerated Docker images (again, see :ref:`containers` for details).
To parallelize across multiple GPUs requires building dynamite from source (see :ref:`installing`); dynamite should then be run with a number of MPI ranks equal to the number of available GPUs.
Achieving good performance with multiple GPUs requires a GPU-aware MPI library.

.. _ranks:

I increased the number of MPI ranks but my computation didn't get faster!
-------------------------------------------------------------------------

Essentially all of dynamite's computations are designed to run in parallel with MPI, and should speed up with more ranks. The main exceptions are 1) computation of reduced density matrices/entanglement entropy and 2) automatic subspace finding via the ``Auto`` subspace class.

.. note::

   Dynamite comes with a built-in benchmarking script, designed to time various computations, as well as initialization etc. Look for ``benchmark.py`` in the ``benchmarking/`` directory of the dynamite git tree. It is also included in the Docker images at ``/home/dnm/benchmarking/benchmark.py``.

Here are a few ideas to explore for why your computation might be slow or not scaling well.

- Each rank needs a sufficient amount of work. If the problem size is too small, the runtime is dominated by startup/initialization costs which don't parallelize. Usually a system size of L=20 or so is enough to see the benefits of parallelization, depending on what computation you're doing.
- Make sure dynamite has access to a number of cores equal to the number of MPI ranks, and that those cores aren't being used heavily by anything else (for example, testing dynamite on a cluster's login node may lead to bad performance if others are, say, compiling software at the same time).
- Often times, dynamite's speed is limited by memory bandwidth, rather than CPU usage (especially if running with multiple ranks that share the same RAM, such as multiple cores on one compute node). A tell-tale sign of this is if the computation speeds up when you go from 1 core to a few, but then stops getting faster as you increase the number of cores in use on the same node (the memory bandwidth has become saturated, so adding more cores doesn't help). In this case, if you are on a cluster, you may want to spread your computation across a few nodes to make full use of each node's memory bandwidth.
- On the other hand, sometimes dynamite may be limited by MPI communication bandwidth, particularly if you are using many nodes on a cluster. This can be improved by trying to use as few nodes as possible (while optimally using the memory bandwidth of each node, see above). There are also various cluster-specific tricks (such as SLURM's ``--switches`` option) that can help you get nodes that are near each other on the cluster's network, allowing for better communication performance.

.. _shell:

My computation is using too much memory.
----------------------------------------

Even in the sparse form that dynamite uses by default, storing an operator's matrix can use large amounts of memory. To alleviate this problem, dynamite can be run with so-called "matrix-free" matrices (known in dynamite and PETSc as "shell" matrices). When this is enabled, matrix elements are computed on the fly instead of being stored explicitly, saving significantly on memory usage and sometimes even speeding things up. When using shell matrices, the memory usage is reduced essentially to the vectors used in the computations.

Shell matrices can be enabled globally by setting ``dynamite.config.shell = True`` at the beginning of your computation, or for a particular operator by setting the ``Operator.shell`` flag to ``True``.

.. _integer:

I got an error message about an integer overflow even though I'm running with fewer than 32 spins.
--------------------------------------------------------------------------------------------------

Even if the state vector length is shorter than :math:`2^{32}`, PETSc may allocate a block of many vectors at once, and the total length of this allocated block is greater than the maximum 32-bit integer. Before switching to 64-bit integers, try passing the ``-bv_type vecs`` flag to SLEPc by putting the following at the beginning of your script:

.. code:: python

    from dynamite import config
    config.initialize(slepc_args=['-bv_type', 'vecs'])

That way each vector will be allocated individually.

.. _L:

I am tired of setting the spin chain length L everywhere.
---------------------------------------------------------

There is an easy way to globally set a
default value for ``L``. Before you start building any operators:

.. code:: python

    from dynamite import config
    config.L = 24  # or whatever you want

There are other global configuration options, too. See the documentation
for details.

.. _nondeterm:

My code is having mysterious problems/giving wrong answers when I run with more than 1 MPI rank.
------------------------------------------------------------------------------------------------

There are a number of reasons this could happen, but here is a likely culprit. Each MPI rank runs as an independent Python process, so non-deterministic code can behave differently across the ranks. For example, if you are iterating through an unordered data type like a Python dictionary or set, different ranks may iterate through the values in a different order! As another example, making calls to e.g. ``numpy.random.rand()`` will give different values on each process. If you use this when building your Hamiltonian, you will not have a consistent operator across your different processes! If you need random numbers, make sure to seed them with the same value everywhere.

.. _petsc:

I want to get under the hood and fiddle with PETSc and SLEPc.
-------------------------------------------------------------

The underlying ``petsc4py`` matrix for any operator is accessible with
:meth:`dynamite.operators.Operator.get_mat`. For states, the ``petsc4py`` vector
is :attr:`dynamite.states.State.vec`. Arbitrary functions from ``petsc4py`` can
be called through this interface. The documentation is not too extensive for
petsc4py and slepc4py, but it is inferred easily from the C interface.
For example, the C function ``VecSum()`` is implemented as a member function of
the Python ``petsc4py.PETSc.Vec`` class: one would just do
``my_state.vec.sum()``. Be careful when using functions that modify the objects however,
as some dynamite internals depend on the ``petsc4py`` objects being in certain states,
and modifying those underlying objects could cause dynamite to break.

The behavior of PETSc and SLEPc can also be modified by certain flags, that would normally
be passed on the command line to C programs using these libraries. These options can be passed
to dynamite as well, via the ``slepc_args`` keyword argument to :meth:`dynamite.config.initialize`.
As an example: to manually change the size of the Krylov subspace used by SLEPc's matrix exponential,
one would do

.. code:: python

    from dynamite import config
    config.initialize(['-mfn_ncv', '40'])

(although this particular case is built-in to dynamite, and can be accomplished via the ``ncv`` keyword
argument to :meth:`dynamite.computations.evolve`).

.. _gpu-aware-mpi:

I am getting a warning from PETSc about not having GPU-aware MPI.
-----------------------------------------------------------------

dynamite is designed to be able to run parallelized across multiple GPUs. For this to be performant,
it is crucial that the MPI implementation being used is GPU-aware, meaning that instead of transferring
data to the CPU, then to another processor via MPI, then to that processor's GPU, it can transfer data
directly between GPUs via e.g. NVLink.

If you are running with multiple GPUs, the way to avoid this error is to ensure your MPI implementation
is GPU-aware---your performance will be quite bad otherwise. If you are compiling OpenMPI yourself, use
the ``--with-cuda`` flag to OpenMPI's ``./configure``; if you are using a compute cluster's build of MPI, talk to
your system administrator.

If you are running with a single GPU, MPI is simply not needed. In that case you can avoid the warning by
removing ``mpi4py`` from your Python environment, in which case dynamite will automatically disable
the warning, or by setting an environment variable as described in the PETSc error message.
