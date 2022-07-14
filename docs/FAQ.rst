
FAQ
===

**I got an error message about an integer overflow even though I'm running with fewer than 32 spins.**

Even if the state vector length is shorter than :math:`2^{32}`, sometimes PETSc allocates a block of many vectors at once, and the total length of this allocated block is greater than the maximum 32-bit integer. Before switching to 64-bit integers, try passing the ``-bv_type vecs`` flag to SLEPc (call ``dynamite.config.initialize(slepc_args=['-bv_type', 'vecs'])`` at the beginning of your script). That way each vector will be allocated individually.

**I am so tired of setting the size of all my matrices to the same value!**

There is an easy way to globally set a
default value for ``L``. Before you start building any operators:

.. code:: python

    from dynamite import config
    config.L = 24  # or whatever you want

There are other global configuration options, too. See the documentation
for details.
