.. _installing:

**********************
Installing from source
**********************

..
   The easiest way to use ``dynamite`` is through the pre-built container images---see :ref:`containers`.
   If for some reason you can't use the containers, or if you want a site-specific build (for example to optimize message passing performance between nodes on a cluster), you can build from source.

The following instructions allow one to build ``dynamite`` manually from source. Experimental support has also been added to use dynamite from a pre-built Docker container; see :ref:`containers` for instructions.

Building from source
====================

.. note ::
    dynamite is written for Python >=3.8! You may need to install an appropriate
    version first if you don't already have it.

Download dynamite
-----------------

.. code:: bash

    git clone https://github.com/GregDMeyer/dynamite.git

Dynamite is built on the `PETSc <www.mcs.anl.gov/petsc/>`_ and `SLEPc <http://slepc.upv.es/>`_
packages. The next step is to install those.

Build PETSc
--------------

To build PETSc in your working directory, as per the
`download page <https://www.mcs.anl.gov/petsc/download/index.html>`_, do the
following. There is a configuration script that comes with dynamite which should help:

.. code:: bash

    git clone --depth 1 --branch v3.17.3 https://gitlab.com/petsc/petsc.git petsc
    cd petsc
    python <dynamite directory>/petsc_config/complex-opt.py

Note that you may want to adjust some of the build options. Just take a look at
the script and modify as desired. There are also a couple other scripts in that
directory for debug builds (if you will be modifying dynamite) and CUDA support.

If all goes well, ``configure`` will tell you to run a ``make`` command. Copy
the command and run it. It should look like:
``make PETSC_DIR=<your_petsc_directory> PETSC_ARCH=complex-opt all``

Building SLEPc
--------------

Before installing SLEPc, make sure to set environment variables describing your
new PETSc installation:

For bash shell:
``export PETSC_DIR=<petsc_dir>; export PETSC_ARCH=complex-opt``

Now download and install SLEPc:

.. code:: bash

    git clone --depth 1 --branch v3.17.1 https://gitlab.com/slepc/slepc.git slepc
    cd slepc
    ./configure

If it configures correctly, it will output a ``make`` command to run. Copy and
paste that, and run it. It should look like:
``make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt``

Building dynamite
-----------------

Dynamite requires Python 3, as well as some packages you can install with pip.
Before you install them, make sure ``PETSC_DIR`` and ``PETSC_ARCH``
environment variables are still set from the above exports (or re-set them). You
should also set ``SLEPC_DIR``:

.. code:: bash

    export SLEPC_DIR=<your_slepc_installation_directory>

I also suggest using a `virtual environment <https://docs.python.org/3/library/venv.html>`_,
to keep all of the packages tidy.

Now, you can install everything by running

.. code:: bash

    pip install $PETSC_DIR/src/binding/petsc4py
    pip install $SLEPC_DIR/src/binding/slepc4py
    cd dynamite
    pip install -r requirements.txt

Finally, install dynamite:

.. code:: bash

    pip install ./

Now you should be all set to use dynamite! If you want to work on the dynamite
source code, or just easily pull updates from GitHub, you might want to do
``pip install -e ./`` to keep the source files in-place.

.. note::

    Don't try to do ``pip install dynamite``! There is a totally unrelated
    package on PyPI by that name.
