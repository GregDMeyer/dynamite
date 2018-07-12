/*
 * This file defines a template for matrix functions. It should be included
 * multiple times in bpetsc_impl.c, with LEFT_SUBSPACE and RIGHT_SUBSPACE defined as
 * the desired values.
 */

// I'm still salty about this double indirection business
#define CONCAT_U(a, b) a ## _ ## b
#define CONCAT_EVAL(a, b) CONCAT_U(a, b)

PetscErrorCode CONCAT_EVAL(ComputeNonzeros, CONCAT_EVAL(LEFT_SUBSPACE, RIGHT_SUBSPACE))
  (PetscInt M, PetscInt N, const msc_t* msc,
   PetscInt** diag_nonzeros, PetscInt** offdiag_nonzeros,
   const void *left_subspace_data, const void *right_subspace_data);

PetscErrorCode CONCAT_EVAL(BuildMat, CONCAT_EVAL(LEFT_SUBSPACE, RIGHT_SUBSPACE))(
  const msc_t *msc,
  const void* left_subspace_data,
  const void* right_subspace_data,
  Mat *A)
{
  PetscErrorCode ierr;
  PetscInt M, N, row_start, row_end, mask_idx, term_idx;
  int mpi_size;
  PetscInt *diag_nonzeros, *offdiag_nonzeros;

  PetscInt row_idx, ket, col_idx, bra, sign;
  PetscScalar value;

  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);

  /* N is dimension of right subspace, M of left */
  M = CONCAT_EVAL(Dim, LEFT_SUBSPACE)(left_subspace_data);
  N = CONCAT_EVAL(Dim, RIGHT_SUBSPACE)(right_subspace_data);

  /* create matrix */
  ierr = MatCreate(PETSC_COMM_WORLD, A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, M, N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);

  /* preallocate memory */
  ierr = CONCAT_EVAL(ComputeNonzeros, CONCAT_EVAL(LEFT_SUBSPACE, RIGHT_SUBSPACE))
          (M, N, msc, &diag_nonzeros, &offdiag_nonzeros,
           left_subspace_data, right_subspace_data);CHKERRQ(ierr);

  if (mpi_size == 1) {
    ierr = MatSeqAIJSetPreallocation(*A, 0, diag_nonzeros);CHKERRQ(ierr);
  }
  else {
    ierr = MatMPIAIJSetPreallocation(*A, 0, diag_nonzeros,
                                     0, offdiag_nonzeros);CHKERRQ(ierr);
  }

  /* compute matrix elements */
  ierr = MatSetOption(*A, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*A, &row_start, &row_end);CHKERRQ(ierr);

  for (row_idx = row_start; row_idx < row_end; ++row_idx) {

    /* each term looks like value*|ket><bra| */
    ket = CONCAT_EVAL(I2S, LEFT_SUBSPACE)(row_idx, left_subspace_data);

    for (mask_idx = 0; mask_idx < msc->nmasks; mask_idx++) {
      bra = ket ^ msc->masks[mask_idx];

      /* sum all terms for this matrix element */
      value = 0;
      for (term_idx = msc->mask_offsets[mask_idx]; term_idx < msc->mask_offsets[mask_idx+1]; ++term_idx) {
        sign = 1 - 2*(__builtin_parity(bra & msc->signs[term_idx]));
        value += sign * msc->coeffs[term_idx];
      }

      col_idx = CONCAT_EVAL(S2I, RIGHT_SUBSPACE)(bra, right_subspace_data);
      ierr = MatSetValues(*A, 1, &row_idx, 1, &col_idx, &value, INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return ierr;

}

#undef  __FUNCT__
#define __FUNCT__ "ComputeNonzeros"
/*
 * Compute the number of diagonal and off-diagonal nonzeros in each row of our matrix.
 * This is used for preallocating memory in which to store the matrix.
 */
PetscErrorCode CONCAT_EVAL(ComputeNonzeros, CONCAT_EVAL(LEFT_SUBSPACE, RIGHT_SUBSPACE))
  (PetscInt M, PetscInt N, const msc_t* msc,
   PetscInt** diag_nonzeros, PetscInt** offdiag_nonzeros,
   const void *left_subspace_data, const void *right_subspace_data)
{
  PetscErrorCode ierr;
  PetscInt mask_idx, row_idx, row_start, col_idx, col_start, state;
  PetscInt local_rows = PETSC_DECIDE;
  PetscInt local_cols = PETSC_DECIDE;
  ierr = PetscSplitOwnership(PETSC_COMM_WORLD, &local_rows, &M);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(PETSC_COMM_WORLD, &local_cols, &N);CHKERRQ(ierr);

  /* prefix sum to get the start indices on each process */
  MPI_Scan(&local_rows, &row_start, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  MPI_Scan(&local_cols, &col_start, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

  /* MPI_Scan includes current value in the sum */
  row_start -= local_rows;
  col_start -= local_cols;

  /* allocate storage for our diagonal and offdiagonal arrays */
  ierr = PetscCalloc1(local_rows, diag_nonzeros);CHKERRQ(ierr);
  ierr = PetscCalloc1(local_rows, offdiag_nonzeros);CHKERRQ(ierr);

  for (row_idx = 0; row_idx < local_rows; row_idx++) {
    state = CONCAT_EVAL(I2S, LEFT_SUBSPACE)(row_idx+row_start, left_subspace_data);
    for (mask_idx = 0; mask_idx < msc->nmasks; ++mask_idx) {
      col_idx = CONCAT_EVAL(S2I, RIGHT_SUBSPACE)(state^msc->masks[mask_idx], right_subspace_data);
      if (col_idx == -1) {
        /* this term is outside the subspace */
        continue;
      }
      else if (col_idx >= col_start && col_idx < col_start+local_cols) {
        (*diag_nonzeros)[row_idx] += 1;
      }
      else {
        (*offdiag_nonzeros)[row_idx] += 1;
      }
    }
  }
  return ierr;
}
