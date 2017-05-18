Dynamite
==



Installing
----
Dynamite is built on the [PETSc](www.mcs.anl.gov/petsc/) and [SLEPc](http://slepc.upv.es/) packages, as well as the Python wrappers around them, `petsc4py` and `slepc4py`. The first step is to install all of those.
#### Building PETSc
To build PETSc in your working directory, do the following:

As per the [download page](https://www.mcs.anl.gov/petsc/download/index.html), do `git clone -b maint https://bitbucket.org/petsc/petsc petsc`.

Then enter the `petsc` directory:
`cd petsc`

And configure for installation:

<pre><code style='white-space:pre-wrap'>./configure --with-scalar-type=complex --with-petsc-arch=complex-opt --with-debugging=0 --with-precision=double --download-mpich --with-fortran-kernels=1 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3</code></pre>

Note that you may want to adjust some of these options---see below for an explanation of each of them.

If all goes well, `configure` will tell you to run a `make` command. Copy the command and run it. It should look like:
`make PETSC_DIR=<your_petsc_directory> PETSC_ARCH=complex-opt all`

If you want, you can run the PETSc tests as specified in the output of `make` (same as above, with `test` in place of `all`).

**Configure options:**

 - `--with-scalar-type=complex` Use complex numbers with PETSc.
 - `--with-petsc-arch=complex-opt` The name of the PETSc build (we call this `PETSC_ARCH` later)
 - `--with-debugging=0` Do not include C debugging symbols, to improve PETSc performance significantly. Since the normal dynamite user won't be messing with the underlying C code, you won't need C debugging.
 - `--with-precision=double` Use double precision floats. This is currently necessary for dynamite, but soon it will be flexible so that one can use either single or double precision.
 - `--download-mpich` If you don't have an MPI implementation, then this downloads and configures `mpich` for PETSc. However, if you do already have MPI set up (for example, supercomputing clusters will for sure already have an implementation), remove this option and configure should find your MPI implementation.
 - `--with-fortran-kernels=1` You may see some speedup from using the complex number kernels written in Fortran rather than C++.
 - `COPTFLAGS=-O3` Optimization flags to pass to the C compiler.
 - `CXXOPTFLAGS=-O3` Opt. flags for C++.
 - `FOPTFLAGS=-O3` Opt. flags for FORTRAN.

 To see all possible options to `configure`, run `./configure --help`. You may want to pipe to `less`; it is a big help page ;)

#### Building SLEPc

Download SLEPc in a similar way:

`git clone -b maint https://bitbucket.org/slepc/slepc`

or you can use the tarball on [their Downloads page](http://slepc.upv.es/download/download.htm).

Then `cd slepc` and set environment variables to describe your PETSc directory. For bash shell:
`export PETSC_DIR=<petsc_dir>`
`export PETSC_ARCH=complex-opt`

Now run `./configure` (no arguments required). If it configures correctly, it will output a `make` command to run. Copy and paste that, and run it. It should look like:
`make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt`

After that runs, it will tell you to test the installation with `make test`. This seems to fail unless you include the installation directories:
`make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt test`

It may suggest running `make install` as well. You don't need to do this if you don't want to. To use dynamite, you can keep it in this local directory.

#### Building dynamite

Dynamite requires Python 3, as well as some packages you can install with pip. These are:

 - `numpy`
 - `cython`
 - `petsc4py`
 - `slepc4py`

These last two are Python wrappers for PETSc and SLEPc. Before you install them, make sure `PETSC_DIR` and `PETSC_ARCH` environment variables are still set from the above exports (or re-set them). Then you should also set `SLEPC_DIR` with `export SLEPC_DIR=<your_slepc_installation_directory>`.

I suggest using a virtual environment, to keep all of the packages tidy. If you don't know what a virtual environment is, google it! They are very useful.

___

**If you use a virtual environment:**

Go to wherever you want your virtual environment stored (maybe wherever you will store your dynamite code).

Then create the virtual environment:
`python3 -m venv $PWD/.venv`

And activate it:
`source .venv/bin/activate` (or run one of the other activate scripts if you do not use bash)

Now `pip` will install things into the virtual environment until you run `deactivate`.

**If you do not use a virtual environment:**

Use `pip` as you normally would, but be aware that when using `pip` with `sudo`, you need to pass the `-E` flag to `sudo` to preserve the environment variables (`PETSC_DIR` etc.). For example, you would do `sudo -E pip3 install petsc4py`.

___

Now to set up the packages:

`pip3 install cython petsc4py slepc4py`
(you may want `sudo -E` depending on your setup, see above)

With those installed, you have to do a bit of building (I will automate this soon):

`cd dynamite/dynamite/backend`

`make backend_impl.o`

`python setup.py -q build_ext --inplace`

Then you should be all set to import dynamite. You can import it directly from the top-level `dynamite` directory, or you can install it.

To install, I would suggest going to the top-level `dynamite` directory and doing `pip install -e ./`. That way you can `git pull` any changes that I make, and they will automatically be integrated (though you might have to run `make build` again).

Using several processes with a Jupyter Notebook
----

