.. _containers:

************************************************
Running dynamite from a container (experimental)
************************************************

These instructions describe how to run ``dynamite`` from a pre-built container image.
If you wish to instead install ``dynamite`` directly, see :ref:`installing`.

.. note::
   Running from containers is currently experimental. Please let us know if you run into any issues or have any suggestions!

On a personal computer, it's easiest to run the container using `podman <https://podman.io/>`_ or `Docker <https://www.docker.com/>`_.
On a shared computer cluster, one can use `Singularity <https://singularity.hpcng.org/>`_, which should come pre-installed in most HPC settings (see your cluster's documentation).


Quick usage guide
=================

Docker/podman on command line
-----------------------------

With Docker or ``podman`` installed (see :ref:`setup` below), run

.. code:: bash

    docker run --rm -it -w /home/dnm/work -v $PWD:/home/dnm/work gdmeyer/dynamite python your_script.py
    # or replace 'docker' with 'sudo podman' if you are using that

A quick explanation of the options:

 - ``--rm -it``: run interactively, and automatically remove the container when finished
 - ``-w /home/dnm/work -v $PWD:/home/dnm/work``: mount the current working directory into the
   container---this lets dynamite see your script! If you need to give dynamite access to
   another directory, be sure to add another ``-v`` command.

.. note::
   If you want to run with multiple processes using MPI, you can simply add ``mpirun -n <np>``
   before ``python`` in the command above. Note that on a cluster, to achieve the best MPI performance
   you should instead build from source (see :ref:`installing`) and use the cluster's native MPI.

Docker Desktop
--------------

.. note::
   TODO: This documentation coming soon!

.. _singularity-usage:

Singularity
-----------

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

.. note ::
    By default, images are cached in ``~/.singularity`` in your home directory, and they can take up a lot of space.
    If your cluster has a "scratch" filesystem, consider adding a line like the following to your ``.bashrc``
    or equivalent, to move the storage location: ``export SINGULARITY_CACHEDIR=<path to scratch>/.singularity``

Jupyter containers
==================

You can use dynamite in JupyterLab, from a container!

**On the command line:**

.. code:: bash

    docker run -p 8888:8888 -v $PWD:/home/dnm/work gdmeyer/dynamite:latest-jupyter
    # or replace 'docker' with 'sudo podman'

Then follow the last link that you see (it should start with ``http://127.0.0.1:8888``).
Your files will be in the ``work`` directory visible in JupyterLab.

**On Docker Desktop:**

.. note::
   TODO: This documentation coming soon!

..
   Run the container ``gdmeyer/dynamite:latest-jupyter`` and follow the link.
   Don't forget to mount a directory in the container so you can save your work.

.. _setup:

Setting up
==========

Linux
-----

You can install either `podman <https://podman.io/getting-started/installation>`_ or Docker to run
the dynamite containers.
Once you have that, you don't need to do anything else---the dynamite image will be downloaded
automatically the first time you run the commands described above!

Mac
---

For M1 Macs, Docker is easier to install and run. 

Windows
-------

.. note::
   TODO: This documentation is coming soon!

Singularity (cluster)
---------------------

Singularity should come preinstalled on most HPC systems (see your cluster's documentation).
To use dynamite, no setup is required---just run the commands given above!
Do read however the note in the section :ref:`singularity-usage` above.


About containers
================

If you've never used a container before, you can think of it as an image of a whole Linux operating system, in which ``dynamite`` and all of its dependencies have already been installed.
On Linux, when you run ``python`` in the container, the ``python`` process runs like a normal process on your computer, but it sees the container's filesystem (where ``dynamite`` is installed) instead of your own.
(On Windows and Mac, the process runs using virtualization).
