
# Example projects

In this directory are several self-contained dynamite projects, which give examples of the kinds of problems dynamite is useful for and some best practices for writing code to solve them.

Look around, and feel free to modify and explore. Each project has its own README which explains the idea and some interesting directions. You may, for example, want to try running some of these on a computer cluster and/or a GPU to see how that affects dynamite's performance! You should also feel free to use these scripts as a jumping off point for your own studies.

## Features demonstrated

Below we give (non-exhaustive) lists of the computations and programming patterns appearing in each of the examples, to help those looking to figure out how to perform particular tasks. If you don't see something listed here, we still suggest looking at the examples, it may just have not made the list below!

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
 - Computing correlation functions

### SYK

 - Mapping Majoranas onto spins
 - Imaginary time evolution
 - Ordinary time evolution
 - The `Parity` subspace
 - Operators that cross subspaces (e.g. change the parity)
 - Out-of-time-order correlators
 - Using quantum typicality to compute operators' thermal expectation values

### Floquet time evolution

 - Time evolution under piecewise time-dependent Hamiltonians
 - Computing entanglement entropy
 - Computing expectation values
 - Tracking each of the above throughout an evolution

## Running in docker

Note that these projects are also included in dynamite's docker images, so you can easily try them out. They are located at `/home/dnm/examples/scripts/`. For example, to run the SYK example [on a GPU in a compute cluster using Singularity](https://dynamite.readthedocs.io/en/latest/containers.html#singularity-usage), just run the following command:

<!-- TODO: make sure this command is right once I have the scripts written! -->
```bash
singularity exec --nv docker://gdmeyer/dynamite:latest-cuda python /home/dnm/examples/scripts/SYK/run_SYK.py
```

## FAQ: the `main()` function

You will notice each of the projects' main script has a `main()` function, and at the bottom has

```python
if __name__ == '__main__':
    main()
```

If you're not familiar with this, it causes `main()` to only run if the script is run directly (like via `python my_script.py`). This allows, for example, one to have another file with `import my_script` that can use functions from `my_script.py` without running the entire computation! This can be useful for testing and debugging. This structure also allows you to encapsulate variables into separate functions that achieve each step of the computation. Organizing your code like this is not at all required for dynamite to work, but we highly suggest it to make your development process easier and to avoid bugs!
