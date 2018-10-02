
FAQ
===

**Why is dynamite crashing with some message about MPI INIT?**

Are you running with ``mpirun``? PETSc and SLEPc are sometimes unhappy if they
are run outside of MPI (see :ref:`parallelism`).

**I tried to run dynamite in a Jupyter Notebook and my kernel died on import. Why?**

This seems to be a result of the same situation as above. Running MPI in a
notebook to keep the libraries happy (and allow use of more than one process) is
possible and pretty cool---just takes a bit of work to set up.

**I am so tired of setting the size of all my matrices to the same value!**

That wasn't a question. But anyway, there is an easy way to globally set a
default value for ``L``. Before you start building any operators:

.. code:: python

    from dynamite import config
    config.L = 24  # or whatever you want

There are other global configuration options, too. See the full documentation
for details.
