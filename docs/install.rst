Installing
==========

.. note ::
    dynamite is written for Python >=3.6! You may need to install an appropriate
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

    git clone https://gitlab.com/petsc/petsc.git petsc
    cd petsc
    git checkout tags/v3.14.1
    python <dynamite directory>/petsc_config/complex-opt.py

Note that you may want to adjust some of the build options. Just take a look at
the script and modify as desired. There are also a couple other scripts in that
directory for debug builds (if you will be modifying dynamite) and CUDA support.

If all goes well, ``configure`` will tell you to run a ``make`` command. Copy
the command and run it. It should look like:
``make PETSC_DIR=<your_petsc_directory> PETSC_ARCH=complex-opt all``

If you want, you can run the PETSc tests as specified in the output of ``make``
(same as above, with ``test`` in place of ``all``).

Building SLEPc
--------------

Before installing SLEPc, make sure to set environment variables describing your
new PETSc installation:

For bash shell:
``export PETSC_DIR=<petsc_dir>; export PETSC_ARCH=complex-opt``

Now download and install SLEPc:

.. code:: bash

    git clone https://gitlab.com/slepc/slepc.git slepc
    cd slepc
    git checkout tags/v3.14.1
    ./configure

If it configures correctly, it will output a ``make`` command to run. Copy and
paste that, and run it. It should look like:
``make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt``

After that runs, it will tell you to test the installation with ``make test``.
This seems to fail unless you include the installation directories:
``make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt test``

It may suggest running ``make install`` as well. You don't need to do this. To
use dynamite, you can keep it in this local directory.

Building dynamite
-----------------

Dynamite requires Python 3, as well as some packages you can install with pip.
These are listed in ``requirements.txt`` in the dynamite root directory. Two of
the packages, ``petsc4py`` and ``slepc4py``, are Python wrappers for PETSc and
SLEPc. Before you install them, make sure ``PETSC_DIR`` and ``PETSC_ARCH``
environment variables are still set from the above exports (or re-set them). You
should also set ``SLEPC_DIR`` with
``export SLEPC_DIR=<your_slepc_installation_directory>``. Then, you can install
everything by just running

.. code:: bash

    cd dynamite
    pip install -r requirements.txt

.. note::
    When using ``pip`` with ``sudo``, you need to pass the ``-E`` flag to
    ``sudo`` to preserve the environment variables (``PETSC_DIR`` etc.).

I suggest using a `virtual environment <https://docs.python.org/3/library/venv.html>`_,
to keep all of the packages tidy.

Finally, install dynamite:

.. code:: bash

    pip install ./  # you may want sudo with pip

Now you should be all set to use dynamite! If you want to work on the dynamite
source code, or just easily pull updates from GitHub, you might want to do
``pip install -e ./`` to keep the source files in-place.

.. note::

    Don't try to do ``pip install dynamite``! There is a totally unrelated
    package on PyPI by that name.
