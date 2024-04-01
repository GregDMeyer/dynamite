
# Changelog

## 0.3.2 - IN PROGRESS

### Added
 - Detailed example scripts (in `examples/scripts`)
 - `Operator.expectation()`, convenience function to compute the expectation value of the operator with respect to a state
 - `dynamite.tools.MPI_COMM_WORLD()` which returns PETSc's MPI communicator object
 - `Operator.precompute_diagonal` flag allows user to tune whether the matrix diagonal should be precomputed and saved, for shell matrices
 - `State.entanglement_entropy` member function (a more convenient way of using `computations.entanglement_entropy`, which also remains)
 - `tools.get_memory_usage` which can measure memory usage on a total, per rank, or per node basis
 - Multi-GPU parallelism via GPU-aware MPI

### Removed
 - `--track_memory` flag to `benchmark.py`---now memory usage is always reported by the benchmarking script
 - `tools.get_max_memory_usage` and `tools.get_cur_memory_usage` in favor of a single function `tools.get_memory_usage`

### Changed
 - `Operator.msc_size` renamed to `Operator.nterms`, and now invokes a call to `Operator.reduce_msc()`
 - shell matrix-vector multiplications are now considerably faster
 - Improved automatic version check; no longer leaves `.dynamite` files in working directory
 - GPU builds now automatically switch to CPU if a GPU is not found (and print a warning)
 - Changed default bind mount location for Docker images to the container user's home directory, `/home/dnm`
 - Renamed some values of `which` argument of `eigsolve()`: `smallest`→`lowest` and `largest`→`highest`
 - Shift-invert ("target") eigsolving on GPU disabled, as PETSc does not support it well

### Fixed
 - Explicit subspace sometimes failed conservation check even when operator was actually conserved
 - Build was broken with Cython 3
 - Work around broken `petsc4py` and `slepc4py` builds with `pip>=23.1` (see [PETSc issue](https://gitlab.com/petsc/petsc/-/issues/1369))
 - `Operator.__str__` and `Operator.table()` were formatted poorly for operators with complex coefficients
 - various issues in `dynamite.extras`
 - Performance was bad on Ampere (e.g. A100) GPUs unless a particular SLEPc flag was set. The flag is now automatically set.

## 0.3.1 - 2023-03-07

### Fixed
 - compiler flags from PETSc were not being applied to the Cython parts of dynamite
 - large memory usage due to a change in PETSc garbage collection

## 0.3.0 - 2023-01-08

### Added
 - Dynamite now automatically checks that an operator is the same across all MPI ranks before building it (thus catching bugs due to e.g. different random number generator seeds on different ranks)
 - `computations.eigsolve()` and `computations.evolve()` now take a `max_its` flag to adjust the solver's iteration limit
 - More descriptive error messages when solvers fail to converge
 - Methods `.scale()`, `.axpy()`, `.scale_and_sum()`, `+`, `+=`, `-`, `-=`, `*`, `*=` for the `states.State` class, allowing states to be summed together and multiplied by scalars
 - More descriptive output when `str()` is called on `states.State` class, and LaTeX representation of states in Jupyter
 - `State.set_all_by_function()` which takes a user-supplied function and applies it to set each element of a state vector
 - `State.set_uniform()` to set the state to a uniform superposition
 - `Operator.infinity_norm()`
 - The `run_all_tests.py` script (in `tests/integration`) now has an optional flag `--test-set <filename>`, which accepts a file listing tests to be run (instead of simply running all available tests)
 - `L` can now be passed to most subspaces as an argument
 - `L` for the `State` class can be inferred from the length of string passed to the `state` argument of the constructor
 - `L` for state class can now be set directly via the `State.L` parameter
 - `XParity` subspace, which implements Parity but in the X basis and can be applied on top of other subspaces

### Changed
 - Order of arguments to `State` class changed
 - Moved `operators.load_from_file` into the Operator class, as `operators.Operator.load`
 - Moved `operators.from_bytes` into the Operator class, as `operators.Operator.from_bytes`
 - Switched to using `pyproject.toml` for package metadata and build configuration
 - `operators.Operator.scale()` no longer returns the operator it's called on, to avoid confusion about whether a new operator is being created
 - Initial product states can be specified with either the characters `U` and `D` or `0` and `1` (previously only `U` and `D` were allowed)
 - `states.State.normalize()` now has no return value (previously returned the scale factor used for normalization)
 - `hash(subspace)` now is only the same for two subspaces if `subspace.identical` would return `True`
 - Subspace methods `idx_to_state` and `state_to_idx` now return a scalar when passed a scalar, instead of returning a length-1 array
 - Removed `spinflip` argument from `SpinConserve` (replaced by `XParity` subspace)

### Fixed
 - An exception is now raised if an unknown keyword argument is passed to `computations.evolve()`
 - Ensure compatibility when saving operators to disk using a 64 bit dynamite install but loading with a 32 bit install
 - Ensure spin indices passed to operator constructors fall within valid range
 - `computations.entanglement_entropy()` and `computations.dm_entanglement_entropy()` sometimes returned bad values due to uninitialized memory being included in the computation
 - Imaginary time evolution is now possible (for entirely real Hamiltonians), when dynamite is built without complex numbers
 - Edge case where matrix norm was incorrect for a few very specific operators, with SpinConserve + spinflip subspace
 - Bug in shell matrix-vector multiply for certain operators when L>31
 - Ensure backend gets recompiled if `PETSC_ARCH` changes
 - `hash(subspace)` is now much more performant for large Hilbert spaces
 - Incorrect memory usage reported by `benchmark.py`
 - Matrix-vector multiplication was incorrect for `Parity` subspace when operator did not conserve the subspace
 - Deprecated `np.float` type

## 0.2.3 - 2022-08-17

### Added
 - `Subspace.identical` method determines whether two subspaces are exactly the same (both equal and of the same type)
 - `Operator.has_subspace` method returns whether a given subspace has been registered with the operator
 - `State.save` and `State.from_file` methods to allow saving states to disk and loading them later
 - `State.initialized` flag to track whether the state's vector has been filled with data
 - `State.set_initialized` and `State.assert_initialized` methods to set and check the flag
 - Expanded Jupyter notebook tutorial
 - `Operator.conserves` method returns whether the given subspace is conserved by the operator
 - When the matrix for an operator is built, subspaces are automatically checked for conservation/compatibility
 - `Operator.allow_projection` property allows the automatic subspace check to be turned off
 - Ability to flag integration tests and skip them based on that flag
 - Automatically disable thread-based parallelism when running with more than one MPI rank

### Changed
 - `L` of operators and states is now stored in their subspaces
 - The value of `L` in a Subspace object now cannot be modified after it is set
 - Moved package source into `src/` directory
 - Operators must share the same value of all settable properties (e.g. L, shell, etc.) in order to be added or multiplied together, rather than ambiguously inheriting one of the differing values
 - Changed default SLEPc solver for computation of time evolution to 'expokit', which seems to be faster and more stable

### Fixed
 - Ensure that the `L` of all of an operator's subspaces remains consistent
 - Ensure that product states passed as an integer to `State.str_to_state` correspond to valid states of the given length L
 - Small fixes and improvements to documentation
 - Do not require `mpi4py` to be installed if running with only 1 rank
 - Creation of Auto subspace is now significantly faster

## 0.2.2 - 2022-07-01

### Added
 - `State.project()` method to project states onto a particular spin value
 - `Operator.truncate()` method to remove small terms
 - `Operator.estimate_memory()` to estimate the memory an operator will use during computation
 - docker builds with 64-bit integers

### Changed
 - default JupyterLab port is now 8887 to avoid conflict with any other running instances of JupyterLab
 - memory usage tracking functions now report results in gigabytes instead of bytes

### Fixed
 - Improvements to docker build process
 - Update deprecated petsc macros in backend
 - broken int64 test

## 0.2.1 - 2022-05-19

### Added

 - Improvements to docker build process
 - Colors in output of integration tests
 - Set desired CUDA compute capability using environment variable, for PETSc config scripts

### Changed
 - Enabled GPU computations by default when dynamite is compiled with GPU support
 - Install petsc4py and slepc4py directly from PETSc/SLEPc source tree instead of PyPI

### Fixed
 - Possible out-of-bounds memory access in the Explicit subspace

## 0.2.0

Beginning of changelog.
