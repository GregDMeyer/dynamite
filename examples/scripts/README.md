
# Example projects

In this directory are several self-contained dynamite projects, which give examples of the kinds of problems dynamite is useful for and some best practices for writing code to solve them.

Look around, and feel free to modify and explore. Each project has its own README which explains the idea and some interesting directions. You may, for example, want to try running some of these on a computer cluster and/or a GPU to see how that affects dynamite's performance! You should also feel free to use these scripts as a jumping off point for your own studies.

## Features demonstrated

Below we give (non-exhaustive) lists of the computations and programming patterns appearing in each of the examples, to help those looking to figure out how to perform particular tasks. If you don't see what you're looking for in these lists, we still suggest looking at the examples; it may just have not made the lists below!

### In all examples

 - Building operators
 - Using `mpi_print` to avoid duplicated output when running with many MPI ranks
 - Separating "data" output and "human" output by printing to `stderr` (except Kagome example)

### MBL

 - Eigensolving for ground states
 - Eigensolving for states in the middle of the spectrum
 - The `SpinConserve` subspace
 - Computing entanglement entropy
 - Coordinating randomness across MPI ranks

### Kagome

 - Building Hamiltonians with arbitrary connectivity
 - Eigensolving for ground states
 - The `SpinConserve` subspace
 - The `XParity` subspace
 - Computing correlation functions (TODO)

### SYK

 - Mapping Majoranas onto spins
 - Imaginary time evolution
 - Ordinary time evolution
 - Generating random states
 - The `Parity` subspace
 - Operators that cross subspaces (in this case, changing the parity)
 - Out-of-time-order correlators
 - Using quantum typicality to compute operators' thermal expectation values
 - Coordinating randomness across MPI ranks

### Floquet time evolution

 - Initializing product states
 - Time evolution under piecewise Hamiltonians
 - Computing entanglement entropy
 - Computing expectation values
 - Tracking each of the above throughout an evolution
 - Checkpointing and restarting from a checkpoint

## Running the scripts

With a working dynamite installation, you can simply `cd` to the example you want to run and do

```bash
python run_<example>.py
```
(for example, `run_mbl.py` for the MBL example).

A key feature of dynamite is the ability to run scripts in parallel via MPI. On most systems you can run (using 4 MPI ranks for example):
```bash
mpirun -n 4 python run_<example>.py
```
On certain systems, like clusters, the command to launch MPI jobs may be different---check your system's documentation!

You can also try running the examples on a GPU, if you have access to one!
If dynamite is compiled with GPU support it will perform computations on the GPU by default. The easiest way to access a dynamite build
that has been compiled with GPU support is probably via the Docker images; see [the documentation](https://dynamite.readthedocs.io/en/latest/containers.html)
for details!

## Running in docker

Note that these examples are included in dynamite's docker images at `/home/dnm/examples/scripts/`, so you can easily try them out. For example, to run the
Kagome example with 21 spins you can simply do

```bash
docker run --rm -it gdmeyer/dynamite:latest python examples/scripts/kagome/run_kagome.py 21
```

Or to run it [on a GPU in a compute cluster using Singularity](https://dynamite.readthedocs.io/en/latest/containers.html#singularity-usage):

```bash
singularity exec --nv docker://gdmeyer/dynamite:latest-cuda python /home/dnm/examples/scripts/kagome/run_kagome.py 21
```

## FAQ: the `main()` function

You will notice each of the projects' run script has a `main()` function, and at the bottom has

```python
if __name__ == '__main__':
    main()
```

If you're not familiar with this, it causes `main()` to only run if the script is run directly (like via `python my_script.py`). This allows, for example, another Python script to do `import my_script` and then use functions from `my_script.py` without running the entire computation! This can be useful for testing and debugging. This structure also allows you to encapsulate variables into separate functions that achieve each step of the computation. Organizing your code like this is not at all required for dynamite to work, but we highly suggest it to make your development process easier and to avoid bugs!
