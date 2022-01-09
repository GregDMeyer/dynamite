.. _containers:

*********************************
Running dynamite from a container
*********************************

On a personal computer, it's easiest to use `podman <https://podman.io/>`_ or `Docker <https://www.docker.com/>`_ to run ``dynamite``.
On a shared computer cluster, one can use `Singularity <https://singularity.hpcng.org/>`_, which should come pre-installed in most HPC settings (see your cluster's documentation).


Quick usage guide
=================

.. note::
   If you want to run with multiple processes using MPI, you can simply add ``mpirun -n <np>``
   before ``python`` in the commands below. Note that on a cluster, to achieve the best MPI performance
   you should instead build from source (see :ref:`installing`) and use the cluster's native MPI.

podman/Docker
-------------

With ``podman`` or Docker installed (see :ref:`setup` below), run

.. code:: bash

    sudo podman run --rm -it -v $PWD:/home/dnm/work docker.io/gdmeyer/dynamite python your_script.py
    # or
    docker run --rm -it -v $PWD:/home/dnm/work gdmeyer/dynamite python your_script.py

A quick explanation of the options:

 - ``--rm -it``: run interactively, and automatically remove the container when finished
 - ``-v $PWD:/home/dnm/work``: mount the current working directory into the container---this lets
   dynamite see your script! If you need to give dynamite access to another directory, be sure to
   add another ``-v`` command.

Docker Desktop
--------------

TODO: test this out!

.. _singularity-usage:

Singularity
-----------

.. note ::
    By default, images are cached in ``~/.singularity`` in your home directory, and they can take up a lot of space.
    If your cluster has a "scratch" filesystem, consider adding a line like the following to your ``.bashrc``
    or equivalent, to move the storage location: ``export SINGULARITY_CACHEDIR=<path to scratch>/.singularity``

To use dynamite on the CPU, you can run

.. code:: bash

    singularity exec docker://gdmeyer/dynamite python your_script.py  # to run a script
    # or
    singularity shell docker://gdmeyer/dynamite  # to start a shell with dynamite installed

The first time you run the command, it will take a while to download and build the image, but after that it should be instantaneous.

If you are on a node with an Nvidia GPU, running the CUDA-accelerated version of dynamite is as easy as:

.. code:: bash

    singularity exec --nv docker://gdmeyer/dynamite:latest-cuda python your_script.py  # to run a script
    # or
    singularity shell --nv docker://gdmeyer/dynamite:latest-cuda  # to start a shell with dynamite installed

.. note ::
   dynamite with CUDA requires Nvidia driver >= 450.80.02


Jupyter containers
==================

You can use dynamite in JupyterLab, from a container!

**On the command line (Linux, macOS):**

.. code:: bash

    sudo podman run -p 8888:8888 -v $PWD:/home/dnm/work docker.io/gdmeyer/dynamite:latest-jupyter
    # or
    docker run -p 8888:8888 -v $PWD:/home/dnm/work gdmeyer/dynamite:latest-jupyter

Then follow the last link that you see (it should start with ``http://127.0.0.1:8888``).

**On Docker Desktop:**

Run the container ``gdmeyer/dynamite:latest-jupyter`` and follow the link.
Don't forget to mount a directory in the container so you can save your work.


.. _setup:

Setting up
==========


podman/Docker (personal computer)
---------------------------------

**Command line (Linux, macOS):**

 1. `Install podman <https://podman.io/getting-started/installation>`_ (or Docker)
 2. ``sudo podman pull docker.io/gdmeyer/dynamite`` or ``docker pull gdmeyer/dynamite``

**Using docker Desktop (Linux, macOS, Windows):**

Docker Desktop is a closed-source application from Docker Inc. which makes it super easy to run Docker containers, if you don't mind signing up for a free account with them.

 1. `Install Docker Desktop <https://www.docker.com/get-started>`_. You can skip the tutorial.
 2. Pull the container ``gdmeyer/dynamite``.
 3. Make sure to set up Docker to mount a directory from your computer into ``/home/dnm/work`` in the container, so you can save and manipulate files.


Singularity (cluster)
---------------------

Singularity should come preinstalled on most HPC systems (see your cluster's documentation).
To use dynamite, no setup is required---just run the commands!
Do read however the note in the section :ref:`singularity-usage` above.


About containers
================

If you've never used a container before, you can think of it as an image of a whole Linux operating system, in which ``dynamite`` and all of its dependencies have already been installed.
On Linux, when you run ``python`` in the container, the ``python`` process runs like a normal process on your computer, but it sees the container's filesystem (where ``dynamite`` is installed) instead of your own.
(On Windows and Mac, the process runs using virtualization).
