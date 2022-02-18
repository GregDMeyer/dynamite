.. _containers:

*********************************
Running dynamite from a container
*********************************

These instructions describe how to run ``dynamite`` from a pre-built container image.
If you wish to instead install ``dynamite`` directly, see :ref:`installing`.

.. note::
   Running from containers is currently experimental. Please let us know if you run into any issues or have any suggestions!

On a personal computer, it's easiest to run the container using `podman <https://podman.io/>`_ or `Docker <https://www.docker.com/>`_.
On a shared computer cluster, one can use `Singularity <https://singularity.hpcng.org/>`_, which should come pre-installed in most HPC settings (see your cluster's documentation).


Quick usage guide
=================

Command line
------------

With Docker or podman installed (see :ref:`setup` below), run

.. code:: bash

    docker run --rm -it -w /home/dnm/work -v $PWD:/home/dnm/work gdmeyer/dynamite python your_script.py
    # or replace 'docker' with 'podman' if you are using that
    # for podman you may need to add "docker.io/" in front of "gdmeyer" in the command

A quick explanation of the options:

 - ``--rm -it``: run interactively, and automatically remove the container when finished
 - ``-w /home/dnm/work -v $PWD:/home/dnm/work``: mount the current working directory into the
   container---this lets dynamite see your script! If you need to give dynamite access to
   another directory, be sure to add another ``-v`` command.

.. note::
   On Windows, you need to replace ``$PWD`` with ``%cd%`` (or whatever Windows path you want to mount
   in the container).

.. note::
   If you want to run with multiple processes using MPI, you can simply add ``mpirun -n <np>``
   before ``python`` in the command above. Note that on a cluster, to achieve the best MPI performance
   you should instead build from source (see :ref:`installing`) and use the cluster's native MPI.
   Also, with Docker you may get errors unless you add the flag ``--cap-add=SYS_PTRACE``.

.. _desktop_script:

Docker Desktop
--------------

Once you have Docker Desktop installed, you can use it from the command line (including in Windows!) as described above.
However, if you prefer to use the GUI, that works fine too.
The only thing you must do from the command line is pull the image: ``docker pull gdmeyer/dynamite:latest``.

Now in the "Images" tab, hover over the dynamite image you just pulled, and hit "Run".
Expand the "Optional Settings" menu, and under "Volumes", mount a directory on your computer ("Host Path") onto ``/home/dnm/work`` in the container ("Container Path").
You can also give the container a name if you want (otherwise Docker will pick a random name for you).
Then hit "Run"!

You will now see your new dynamite container running under the "Containers/Apps" tab of Docker Desktop.
Hover over it, and click the "CLI" button.
This will execute a ``sh`` shell in the container (you may want to run the ``bash`` command in this shell, to get a bit more useable shell).
Now you can interact with it like any Linux system with dynamite installed.

.. _singularity-usage:

Singularity
-----------

To use dynamite on a cluster with Singularity installed, you can run

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

Command line
------------

.. code:: bash

    docker run --rm -p 8888:8888 -w /home/dnm/work -v $PWD:/home/dnm/work gdmeyer/dynamite:latest-jupyter
    # or replace 'docker' with 'podman'

Then follow the last link that you see (it should start with ``http://127.0.0.1:8888``).
Your files will be in the ``work`` directory visible in JupyterLab.

Docker Desktop
--------------

Follow the steps described above in `Docker Desktop <#docker-desktop>`_, but pull and use the ``gdmeyer/dynamite:latest-jupyter`` container instead of just ``gdmeyer/dynamite``.
Also, perform the following extra steps:

 - During the setup phase, in "Optional Settings" type ``8888`` in "Local Host" to bind port 8888 in the container to port 8888 on your host machine
 - The first time, you may need to allow Docker through the Windows firewall
 - Once the container is running, click on it (anywhere) to view "Logs", and then follow the last link in the output (the one that starts with ``http://127.0.0.1:8888``). You can also click the "Open in Browser", but you will need to find the access token in the logs.

On a compute cluster
--------------------

You can also run the Jupyter notebook containers on a compute cluster, via singularity!
This can allow you to leverage the power of the cluster (including GPUs) in a notebook.
It may take some tweaking for your specific compute cluster, but the basic steps are:

 1. Login, and allocate a compute node for yourself on the cluster (e.g. with ``salloc`` in SLURM).
 2. In a separate terminal, tunnel port 8888 to your local machine through ssh:
    - Run ``ssh -NL 8888:<hostname of compute node from step 1>:8888 <username>@<cluster login url>``
    - The above command should not generate any output
 3. On the compute node from Step 1, run ``singularity run docker://gdmeyer/dynamite:latest-jupyter``
 4. Follow the last link in the output (the one with ``127.0.0.1``)

If you have a GPU on your compute node, you can add the ``--nv`` flag to the singularity command and use the ``gdmeyer/dynamite:latest-cuda-jupyter`` container (see :ref:`singularity-usage` section above).

.. _setup:

Setting up
==========

Linux
-----

You can either `install podman <https://podman.io/getting-started/installation>`_ or
`install Docker <https://docs.docker.com/engine/install/#server>`_ to run the dynamite containers.
Once you have that, you don't need to do anything else---the dynamite image will be downloaded
automatically the first time you run the commands described above!

Mac + Windows
-------------

It is easiest to install Docker via Docker Desktop. Simply `install Docker Desktop <https://www.docker.com/products/docker-desktop>`_. With that installed, just run the commands above using the Mac terminal or Windows command line.

On Windows, you may need to install some Windows Subsystem for Linux components---Docker should guide you through it.

If you are particularly inclined towards open source, you may want to run the containers `using podman instead <https://podman.io/getting-started/installation#windows>`_.

Singularity (cluster)
---------------------

Singularity should come preinstalled on most HPC systems (see your cluster's documentation).
To use dynamite, no setup is required---just run the commands given above!
Do read however the note in the section :ref:`singularity-usage` above.

Alternatively, the cluster may use Shifter to run containers---see your cluster's documentation.

Installing other packages in your container
===========================================

If you want to install other Python packages or other software to use alongside dynamite, it is possible to do this with Docker.
However, it's a little annoying; if the extra software is for analysis or similar we recommend saving the output of your dynamite computation to a file in your mounted directory (e.g. ``/home/dnm/work``) and then performing the analysis after-the-fact.

A quick explainer of what's happening here: when you run dynamite using the commands in the `Quick Usage Guide`_ section above, Docker creates a "container" on top of the dynamite image.
With the ``--rm`` flag as described above, this container is simply removed when the program run inside docker exits.
However, by removing the ``--rm`` flag (and perhaps adding a ``--name``), we can keep the container around, make changes, add things, etc.

So, to make a persistent container, which mounts the current directory at ``/home/dnm/work``, run dynamite like this:

.. code:: bash

    docker run --name my_dnm_container -it -v $PWD:/home/dnm/work gdmeyer/dynamite bash

This will give you a bash shell, where you can run ``pip install <whatever>`` or anything else you would like.
Note that the directory mount (the ``-v`` option) is a part of the container, so when you run the commands below the same directory will always be mounted at ``/home/dnm/work``.

After you exit the bash shell above, the next time you want to use the same container, run

.. code:: bash

    docker start my_dnm_container

Now the container is running, and you can do arbitrary commands in it with ``docker exec``. For example:

.. code:: bash

    # all of the following commands will work now
    docker exec my_dnm_container python my_script.py
    docker exec -it my_dnm_container bash
    docker exec my_dnm_container pip install matplotlib

where the ``-it`` makes the session interactive.
Note that ``docker exec`` just spawns a new process in the container---so you can have potentially many things running at the same time in the same container using this command.

About containers
================

If you've never used a container before, you can think of it as an image of a whole Linux operating system, in which ``dynamite`` and all of its dependencies have already been installed.
On Linux, when you run ``python`` in the container, the ``python`` process runs like a normal process on your computer, but it sees the container's filesystem (where ``dynamite`` is installed) instead of your own.
(On Windows and Mac, the process runs using virtualization).
