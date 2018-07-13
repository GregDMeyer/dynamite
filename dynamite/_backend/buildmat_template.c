/*
 * This file defines a template for matrix functions. It should be included
 * multiple times in bpetsc_impl.c, with LEFT_SUBSPACE and RIGHT_SUBSPACE defined as
 * the desired values.
 */

#include "buildmat_template.h"

#undef  __FUNCT__
#define __FUNCT__ "BuildMat"
PetscErrorCode C(BuildMat,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const void* left_subspace_data,
  const void* right_subspace_data,
  shell_impl shell,
  Mat *A)
{
  PetscErrorCode ierr;
  if (shell == NO_SHELL) {
    ierr = C(BuildPetsc,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
      msc, left_subspace_data, right_subspace_data, A);
  }
  else if (shell == CPU_SHELL) {
    ierr = C(BuildCPUShell,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
      msc, left_subspace_data, right_subspace_data, A);
  }
  /* TODO */
  // else if (shell == GPU_SHELL) {
  //   ierr = BuildGPUShell(msc, left_subspace_data, right_subspace_data, A);
  // }
  else {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Invalid shell implementation type.");
  }
  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "BuildPetsc"
PetscErrorCode C(BuildPetsc,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
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
  M = C(Dim,LEFT_SUBSPACE)(left_subspace_data);
  N = C(Dim,RIGHT_SUBSPACE)(right_subspace_data);

  /* create matrix */
  ierr = MatCreate(PETSC_COMM_WORLD, A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, M, N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);

  /* preallocate memory */
  ierr = C(ComputeNonzeros,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))
          (M, N, msc, &diag_nonzeros, &offdiag_nonzeros,
           left_subspace_data, right_subspace_data);CHKERRQ(ierr);

  if (mpi_size == 1) {
    ierr = MatSeqAIJSetPreallocation(*A, 0, diag_nonzeros);CHKERRQ(ierr);
  }
  else {
    ierr = MatMPIAIJSetPreallocation(*A, 0, diag_nonzeros,
                                     0, offdiag_nonzeros);CHKERRQ(ierr);
  }

  /* this memory is allocated in ComputeNonzeros */
  ierr = PetscFree(diag_nonzeros);
  ierr = PetscFree(offdiag_nonzeros);

  /* compute matrix elements */
  ierr = MatSetOption(*A, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*A, &row_start, &row_end);CHKERRQ(ierr);

  for (row_idx = row_start; row_idx < row_end; ++row_idx) {

    /* each term looks like value*|ket><bra| */
    ket = C(I2S,LEFT_SUBSPACE)(row_idx, left_subspace_data);

    for (mask_idx = 0; mask_idx < msc->nmasks; mask_idx++) {
      bra = ket ^ msc->masks[mask_idx];

      /* sum all terms for this matrix element */
      value = 0;
      for (term_idx = msc->mask_offsets[mask_idx]; term_idx < msc->mask_offsets[mask_idx+1]; ++term_idx) {
        sign = 1 - 2*(__builtin_parity(bra & msc->signs[term_idx]));
        value += sign * msc->coeffs[term_idx];
      }

      col_idx = C(S2I,RIGHT_SUBSPACE)(bra, right_subspace_data);
      ierr = MatSetValues(*A, 1, &row_idx, 1, &col_idx, &value, INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return ierr;

}

#undef  __FUNCT__
#define __FUNCT__ "BuildCPUShell"
PetscErrorCode C(BuildCPUShell,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const void* left_subspace_data,
  const void* right_subspace_data,
  Mat *A)
{
  PetscErrorCode ierr;
  PetscInt M, N, m, n;
  shell_context *ctx;

  /* N is dimension of right subspace, M of left */
  M = C(Dim,LEFT_SUBSPACE)(left_subspace_data);
  N = C(Dim,RIGHT_SUBSPACE)(right_subspace_data);

  m = PETSC_DECIDE;
  n = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD,&m,&M);
  PetscSplitOwnership(PETSC_COMM_WORLD,&n,&N);

  // TODO: for Auto, make a copy of the subspace data

  ierr = BuildContext(msc, left_subspace_data, right_subspace_data, &ctx);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, ctx, A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_MULT,
           (void(*)(void))C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE)));
  ierr = MatShellSetOperation(*A, MATOP_NORM,
           (void(*)(void))C(MatNorm_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE)));

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "ComputeNonzeros"
/*
 * Compute the number of diagonal and off-diagonal nonzeros in each row of our matrix.
 * This is used for preallocating memory in which to store the matrix.
 */
PetscErrorCode C(ComputeNonzeros,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))
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
    state = C(I2S,LEFT_SUBSPACE)(row_idx+row_start, left_subspace_data);
    for (mask_idx = 0; mask_idx < msc->nmasks; ++mask_idx) {
      col_idx = C(S2I,RIGHT_SUBSPACE)(state^msc->masks[mask_idx], right_subspace_data);
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

/*
 * MatMult for CPU shell matrices.
 */
PetscErrorCode C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  PetscErrorCode ierr;
  return ierr;
}

/*
 * MatNorm for CPU shell matrices.
 */
PetscErrorCode C(MatNorm_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  Mat A, NormType type, PetscReal *nrm)
{
  PetscErrorCode ierr;
  PetscInt row_idx, row_start, row_end;
  PetscInt mask_idx, term_idx, ket, bra;
  PetscInt sign;
  PetscScalar csum;
  PetscReal sum, sum_err, comp, total, local_max, global_max;
  shell_context *ctx;

  if (type != NORM_INFINITY) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Only NORM_INFINITY is implemented for shell matrices.");
  }

  ierr = MatShellGetContext(A, &ctx);CHKERRQ(ierr);

  /*
   * keep the norm cached so we don't have to compute it all the time.
   * if we already have it, just return it
   */
  if (ctx->nrm != -1) {
    (*nrm) = ctx->nrm;
    return ierr;
  }

  ierr = MatGetOwnershipRange(A, &row_start, &row_end);CHKERRQ(ierr);

  local_max = 0;
  for (row_idx = row_start; row_idx < row_end; ++row_idx) {

    ket = C(I2S,LEFT_SUBSPACE)(row_idx, ctx->left_subspace_data);

    /* sum abs of all matrix elements in this row */
    /* for precision reasons use the Kahan summation algorithm */
    sum = 0;
    sum_err = 0;
    for (mask_idx = 0; mask_idx < ctx->nmasks; ++mask_idx) {

      bra = ket ^ ctx->masks[mask_idx];

      /* sum all terms for this matrix element */
      csum = 0;
      for (term_idx = ctx->mask_offsets[mask_idx];
           term_idx < ctx->mask_offsets[mask_idx+1];
           ++term_idx) {
        sign = 1 - 2*(__builtin_parity(bra & ctx->signs[term_idx]));
        if (TERM_REAL(ctx->masks[mask_idx], ctx->signs[term_idx])) {
          csum += sign * ctx->real_coeffs[term_idx];
        }
        else {
          csum += I * (sign * ctx->real_coeffs[term_idx]);
        }
      }

      comp = PetscAbsComplex(csum) - sum_err;
      total = sum + comp;
      sum_err = (total - sum) - comp;
      sum = total;
    }

    if (sum > local_max) {
      local_max = sum;
    }
  }

  ierr = MPIU_Allreduce(&local_max, &global_max, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD);CHKERRQ(ierr);

  ctx->nrm = global_max;
  (*nrm) = global_max;

  return ierr;
}
