
// I'm still salty about this double indirection business
#define CONCAT_U(a, b) a ## _ ## b
#define C(a, b) CONCAT_U(a, b)

/*
 * This function is called to build any matrix.
 */
PetscErrorCode C(BuildMat,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const void* left_subspace_data,
  const void* right_subspace_data,
  shell_impl shell,
  Mat *A);

/*
 * Build a standard PETSc matrix.
 */
PetscErrorCode C(BuildPetsc,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const void* left_subspace_data,
  const void* right_subspace_data,
  Mat *A);

/*
 * Build a CPU shell matrix.
 */
PetscErrorCode C(BuildCPUShell,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const void* left_subspace_data,
  const void* right_subspace_data,
  Mat *A);

/*
 * Compute the number of nonzeros per row, for memory allocation purposes.
 */
PetscErrorCode C(ComputeNonzeros,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  PetscInt M, PetscInt N, const msc_t* msc,
  PetscInt** diag_nonzeros, PetscInt** offdiag_nonzeros,
  const void *left_subspace_data, const void *right_subspace_data);

/*
 * MatMult for CPU shell matrices.
 */
PetscErrorCode C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b);

/*
 * MatNorm for CPU shell matrices.
 */
PetscErrorCode C(MatNorm_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  Mat A, NormType type, PetscReal *nrm);
