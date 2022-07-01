
# Changelog

## 0.2.2 - 2022-07-01

### Added
 - `State.project()` method to project states onto a particular spin value
 - `Operator.truncate()` method to remove small terms
 - `Operator.estimate_memory()` to estimate the memory an operator will use during computation
 - docker builds with 64-bit integers

### Changed
 - default JupyterLab port is now 8887 to avoid conflict with any other running instances of JupyterLab

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
