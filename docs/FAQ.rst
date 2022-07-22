
FAQ
===

- :ref:`ranks`
- :ref:`integer`
- :ref:`L`

.. _ranks:

I increased the number of MPI ranks but my computation didn't get faster!
-------------------------------------------------------------------------

Essentially all of dynamite's computations are designed to run in parallel with MPI, and should speed up with more ranks. The main exceptions are 1) computation of reduced density matrices/entanglement entropy and 2) automatic subspace finding via the ``Auto`` subspace class.

.. note::

   Dynamite comes with a built-in benchmarking script, designed to time various computations, as well as initialization etc. Look for ``benchmark.py`` in the ``benchmarking/`` directory of the dynamite git tree. It is also included in the Docker images at ``/home/dnm/benchmarking/benchmark.py``.

Here are a few ideas to explore for why your computation might be slow or not scaling well.

- Each rank needs a sufficient amount of work. If the problem size is too small, the runtime is dominated by startup/initialization costs which don't parallelize. Usually a system size of L=20 or so is enough to see the benefits of parallelization, depending on what computation you're doing.
- Try setting the environment variable ``export OMP_NUM_THREADS=1`` before running. Sometimes OpenMP tries to parallelize at the same time as MPI does, and they use each others' resources which slows things down.
- Make sure dynamite has access to a number of cores equal to the number of MPI ranks, and that those cores aren't being used heavily by anything else (for example, testing dynamite on a cluster's login node may lead to bad performance if others are, say, compiling software at the same time).
- Often times, dynamite's speed is limited by memory bandwidth, rather than CPU usage (especially if running with multiple ranks that share the same RAM, such as multiple cores on one compute node). A tell-tale sign of this is if the computation speeds up when you go from 1 core to a few, but then stops getting faster as you increase the number of cores in use on the same node (the memory bandwidth has become saturated, so adding more cores doesn't help). In this case, if you are on a cluster, you may want to spread your computation across a few nodes to make full use of each node's memory bandwidth.
- On the other hand, sometimes dynamite may be limited by MPI communication bandwidth, particularly if you are using many nodes on a cluster. This can be improved by trying to use as few nodes as possible (while optimally using the memory bandwidth of each node, see above). There are also various cluster-specific tricks (such as SLURM's ``--switches`` option) that can help you get nodes that are near each other on the cluster's network, allowing for better communication performance.

.. _integer:

**I got an error message about an integer overflow even though I'm running with fewer than 32 spins.**
------------------------------------------------------------------------------------------------------

Even if the state vector length is shorter than :math:`2^{32}`, sometimes PETSc allocates a block of many vectors at once, and the total length of this allocated block is greater than the maximum 32-bit integer. Before switching to 64-bit integers, try passing the ``-bv_type vecs`` flag to SLEPc (call ``dynamite.config.initialize(slepc_args=['-bv_type', 'vecs'])`` at the beginning of your script). That way each vector will be allocated individually.

.. _L:

**I am tired of setting the spin chain length L everywhere.**
---------------------------------------------------------------------------

There is an easy way to globally set a
default value for ``L``. Before you start building any operators:

.. code:: python

    from dynamite import config
    config.L = 24  # or whatever you want

There are other global configuration options, too. See the documentation
for details.
