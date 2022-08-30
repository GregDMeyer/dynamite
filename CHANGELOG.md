
# Changelog

## 0.2.4 - IN PROGRESS

### Added

### Changed
 - Moved `operators.load_from_file` into the Operator class, as `operators.Operator.load`
 - Moved `operators.from_bytes` into the Operator class, as `operators.Operator.from_bytes`

### Fixed
 - An exception is now raised if an unknown keyword argument is passed to `computations.evolve()`
 - Ensure compatibility when saving operators to disk using a 64 bit dynamite install but loading with a 32 bit install
 - Ensure spin indices passed to operator constructors fall within valid range

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
