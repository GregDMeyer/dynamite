
Tips, Tricks, and Pitfalls
==========================

Pitfalls (TL,DR)
----------------
 - You may get odd behavior if you do not run under MPI (see Parallelism below).
 - Just because it's easy to build matrices in dynamite doesn't mean you should do it unnecessarily. For example, to compute :math:`\left< \sigma^x_0 \right>`, try to avoid doing ``state.dot(Sigmax(0,L=L)*state)`` repeatedly---it will rebuild the matrix every time. Better to put ``Sigmax(0,L=L)`` in a variable and then reuse it.
 - When working with large spin chains, one is often memory-limited. So, it is good to reuse memory that's already been allocated. :meth:`dynamite.computations.evolve` has a ``result`` argument for exactly this purpose---to put the result into a vector that's already been allocated, instead of allocating a new one. Garbage collection works pretty well, so stuff won't be just accumulating in memory, but it's better and faster to reuse memory explicitly.
 - The algorithm for finding eigenvalues is really good at finding a few eigenvalues in the spectrum. It is not good at finding the entire spectrum. If you try to find the whole spectrum, you will not find any more success than if you used other methods.
 - It really is useful to read the SLEPc (and PETSc) Users' Manual!

.. _parallelism:

Parallelism
-----------

PETSc and SLEPc are built to leverage massively parallel computing. They use the MPI (message-passing interface) framework to communicate between processes. Accordingly, dynamite should be run with MPI. Even when using just one process, it is usually happier to be run under MPI. To do so (the ``-n`` flag specifies the number of processes):

.. code:: bash

    mpirun -n 4 python3 solve_all_the_things.py

Or, if you have MPI installed in an unusual way (e.g. PETSc installed it for you), you may want:

.. code:: bash

    $PETSC_DIR/bin/petscmpiexec -n 4 python3 solve_quantum_gravity.py

Note that there is a tradeoff in parallel processing. Using more processes breaks up the problem, so each process has less to compute, which in general makes the program run faster. However, communication between processes takes a significant amount of time when the number of processes gets too large.

Shell matrices
--------------

Shell, or "matrix-free" matrices, save significantly on memory usage while taking some cost in run time. Instead of storing the entire matrix in memory, they compute matrix elements on-the-fly when they are needed. When using shell
matrices with a reasonably sparse Hamiltonian, the only significant memory usage is the storage of the state vector (and any other vectors used in the evolution or eigensolve computations). Details of dynamite's shell matrix implementation will be documented soon.

Shell matrices can be used easily and selectively in dynamite. For some operator O, one just has to do:

.. code:: python

    O.use_shell = True

They don't have to be used just for the Hamiltonian, either---in fact, one of their best use cases is for finding
expectation values of simple operators. That takes a negligible amount of time anyway, so it is silly to store the whole matrix in memory. For example, to find each :math:`\left< S^z_i \right>`:

.. code:: python

    Szs = []
    for i in range(L):
        Szs.append(Sigmaz(i,L=L))
        Szs[i].use_shell = True

    for state in interesting_states:
        for i in range(L):
            sz_exp = state.dot(Szs[i]*state)

Note that the ``Szs[i]*state`` expression will allocate a new vector each time, which might be OK but perhaps not if one is really pushing the Hilbert space size. One can do slightly better at some cost of readability by changing that last bit to:

.. code:: python

    r = interesting_states[0].copy()
    for state in interesting_states:
        for i in range(L):
            # multiply Szs[i]*state and put it in r
            Szs[i].get_mat().mult(state,r)

            sz_exp = state.dot(r)

That way the memory for the result is only allocated once. For an explanation of the ``get_mat().mult()`` line, see "Interacting with PETSc/SLEPc" below.

Jupyter Notebook Integration
----------------------------

dynamite integrates well with Jupyter Notebooks, outputting the form of operators in beautiful TeX representation inline. However, getting MPI set up under Jupyter is a bit of a hassle, and running without MPI can lead to odd
behavior as mentioned above under Parallelism. A guide to setting it up is coming soon.

Interacting with PETSc/SLEPc
----------------------------

The underlying PETSc matrix for any operator is easily accessible with :meth:`dynamite.operators.Operator.get_mat`, and states in dynamite are just petsc4py vectors themselves. So arbitrary functions from petsc4py can be used with them. The documentation is not too extensive for the petsc4py and slepc4py, but it is inferred easily from the C interface. For example, the C function ``MatMult()`` is implemented as a member function of the Python :meth:`petsc4py.PETSc.Mat` class: one would just do ``my_matrix.mult(in_vec,result_vec)``.

C programs using PETSc and SLEPc can take options at runtime that modify how the libraries work. This is possible in dynamite as well. It is accomplished by using :meth:`dynamite.config.initialize`. An example: if PETSc is configured with GPU support, the following will cause the computations to be performed on the GPU:

.. code:: python

    # at very beginning of script
    from dynamite import config
    config.initialize(['-vec_type','cuda',
                       '-mat_type','aijcusparse',
                       '-bv_type','vecs'])

Note that this must be called at the very start of a script. Calling, for example, ``from dynamite.operators import *`` will automatically initialize PETSc with no arguments.

GPU Support
-----------

It is possible to run dynamite computations on GPUs, and it is amazingly fast. However, it can be a headache to set up (it requires using the development branch of PETSc/SLEPc). Good luck if you try to do so, it's worth it when it works! To get you started, see the `PETSc page on GPU support <https://www.mcs.anl.gov/petsc/features/gpus.html>`_.