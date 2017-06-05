Installing
==========

Dynamite is built on the `PETSc <www.mcs.anl.gov/petsc/>`_ and `SLEPc <http://slepc.upv.es/>`_ packages, as well as the Python wrappers around them, ``petsc4py`` and ``slepc4py``. The first step is to install those.

Building PETSc
--------------

To build PETSc in your working directory, as per the `download page <https://www.mcs.anl.gov/petsc/download/index.html>`_, do the following:

.. code:: bash

    git clone -b maint https://bitbucket.org/petsc/petsc petsc
    cd petsc
    ./configure --with-scalar-type=complex --with-petsc-arch=complex-opt \
    --with-debugging=0 --download-mpich --with-fortran-kernels=1 \
    COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3

Note that you may want to adjust some of these options---see below for an explanation of each of them.

If all goes well, ``configure`` will tell you to run a ``make`` command. Copy the command and run it. It should look like:
``make PETSC_DIR=<your_petsc_directory> PETSC_ARCH=complex-opt all``

If you want, you can run the PETSc tests as specified in the output of ``make`` (same as above, with ``test`` in place of ``all``).

**Configure options:**

 - ``--with-scalar-type=complex`` Use complex numbers with PETSc.
 - ``--with-petsc-arch=complex-opt`` The name of the PETSc build (we call this ``PETSC_ARCH`` later)
 - ``--with-debugging=0`` Do not include C debugging symbols, to improve PETSc performance significantly. Since the normal dynamite user won't be messing with the underlying C code, you won't need C debugging.
 - ``--download-mpich`` If you don't have an MPI implementation, then this downloads and configures ``mpich`` for PETSc. However, if you do already have MPI set up (for example, supercomputing clusters will for sure already have an implementation), remove this option and configure should find your MPI implementation.
 - ``--with-fortran-kernels=1`` You may see some speedup from using the complex number kernels written in Fortran rather than C++.
 - ``COPTFLAGS=-O3`` Optimization flags to pass to the C compiler.
 - ``CXXOPTFLAGS=-O3`` Opt. flags for C++.
 - ``FOPTFLAGS=-O3`` Opt. flags for FORTRAN.

 Optional:

 - ``--use-64-bit-indices`` Required to work with spin chains longer than 31.

 To see all possible options to ``configure``, run ``./configure --help``. You may want to pipe to ``less``; it is a big help page ;)

Building SLEPc
--------------

Before installing SLEPc, make sure to set environment variables describing your new PETSc installation:

For bash shell:
``export PETSC_DIR=<petsc_dir>; export PETSC_ARCH=complex-opt``

Now download and install SLEPc:

.. code:: bash

    git clone -b maint https://bitbucket.org/slepc/slepc
    cd slepc
    ./configure

If you want, instead of git, you can use the tarball on `their Downloads page <http://slepc.upv.es/download/download.htm>`_.

If it configures correctly, it will output a ``make`` command to run. Copy and paste that, and run it. It should look like:
``make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt``

After that runs, it will tell you to test the installation with ``make test``. This seems to fail unless you include the installation directories:
``make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt test``

It may suggest running ``make install`` as well. You don't need to do this. To use dynamite, you can keep it in this local directory.

Building dynamite
-----------------

Dynamite requires Python 3, as well as some packages you can install with pip. These are:

 - ``numpy``
 - ``cython``
 - ``petsc4py``
 - ``slepc4py``

These last two are Python wrappers for PETSc and SLEPc. Before you install them, make sure ``PETSC_DIR`` and ``PETSC_ARCH`` environment variables are still set from the above exports (or re-set them). Then you should also set ``SLEPC_DIR`` with ``export SLEPC_DIR=<your_slepc_installation_directory>``.

.. note::
    When using ``pip`` with ``sudo``, you need to pass the ``-E`` flag to ``sudo`` to preserve the environment variables (``PETSC_DIR`` etc.). For example, you would do ``sudo -E pip3 install petsc4py``.

I suggest using a `virtual environment <https://docs.python.org/3/library/venv.html>`_, to keep all of the packages tidy.

Now to set up the packages:

``pip3 install cython petsc4py slepc4py``
(you may want ``sudo -E`` depending on your setup, see above)

With those installed, you have to do a bit of building (I will automate this soon):

.. code:: bash

    cd dynamite/dynamite/backend
    make backend_impl.o
    python setup.py -q build_ext --inplace

Then you should be all set to import dynamite. You can import it directly from the top-level ``dynamite`` directory, or you can install it by doing ``pip install ./`` in the top-level directory.
