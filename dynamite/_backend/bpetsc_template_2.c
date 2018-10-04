/*
 * This file defines a template for matrix functions. It should be included
 * multiple times in bpetsc_impl.c, with LEFT_SUBSPACE and RIGHT_SUBSPACE defined as
 * the desired values.
 */

#include "bpetsc_template_2.h"
#if PETSC_HAVE_VECCUDA
#include "bcuda_template.h"
#endif

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
#if PETSC_HAVE_VECCUDA
  else if (shell == GPU_SHELL) {
    ierr = C(BuildGPUShell,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
      msc, left_subspace_data, right_subspace_data, A);
  }
#endif
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
  PetscInt M, N, row_start, row_end, col_start;
  PetscInt mask_idx, term_idx, row_count;
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

  /* TODO: we only should call these preallocation routines if matrix type is aij */
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
  ierr = MatGetOwnershipRangeColumn(*A, &col_start, NULL);CHKERRQ(ierr);

  for (row_idx = row_start; row_idx < row_end; ++row_idx) {

    /* each term looks like value*|ket><bra| */
    ket = C(I2S,LEFT_SUBSPACE)(row_idx, left_subspace_data);

    row_count = 0;
    for (mask_idx = 0; mask_idx < msc->nmasks; mask_idx++) {
      bra = ket ^ msc->masks[mask_idx];

      col_idx = C(S2I,RIGHT_SUBSPACE)(bra, right_subspace_data);
      if (col_idx == -1) {
        continue;
      }

      /* sum all terms for this matrix element */
      value = 0;
      for (term_idx = msc->mask_offsets[mask_idx]; term_idx < msc->mask_offsets[mask_idx+1]; ++term_idx) {
        sign = 1 - 2*(builtin_parity(bra & msc->signs[term_idx]));
        value += sign * msc->coeffs[term_idx];
      }

      row_count++;
      ierr = MatSetValue(*A, row_idx, col_idx, value, INSERT_VALUES);CHKERRQ(ierr);
    }

    /* workaround for a bug in PETSc that triggers if there are empty rows */
    if (row_count == 0) {
      ierr = MatSetValue(*A, row_idx, col_start, 0, INSERT_VALUES);CHKERRQ(ierr);
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
    /* as part of workaround for PETSc bug (see BuildPetsc), need at least one element in each row */
    if ((*diag_nonzeros)[row_idx] == 0) {
      (*diag_nonzeros)[row_idx] = 1;
    }
  }
  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "BuildCPUShell"
PetscErrorCode C(BuildCPUShell,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const C(data,LEFT_SUBSPACE)* left_subspace_data,
  const C(data,RIGHT_SUBSPACE)* right_subspace_data,
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

  ierr = C(BuildContext_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
    msc, left_subspace_data, right_subspace_data, &ctx);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, ctx, A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A, MATOP_MULT,
           (void(*)(void))C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE)));
  ierr = MatShellSetOperation(*A, MATOP_NORM,
           (void(*)(void))C(MatNorm_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE)));
  ierr = MatShellSetOperation(*A, MATOP_DESTROY,
           (void(*)(void))C(MatDestroyCtx_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE)));

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "BuildContext_CPU"
/*
 * Build the shell context.
 */
PetscErrorCode C(BuildContext_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const C(data,LEFT_SUBSPACE)* left_subspace_data,
  const C(data,RIGHT_SUBSPACE)* right_subspace_data,
  shell_context **ctx_p)
{
  PetscErrorCode ierr;
  shell_context *ctx;
  PetscInt nterms, i;
  PetscReal real_part;

  ierr = PetscMalloc1(1, ctx_p);CHKERRQ(ierr);
  ctx = (*ctx_p);

  ctx->nmasks = msc->nmasks;
  ctx->nrm = -1;
  nterms = msc->mask_offsets[msc->nmasks];

  /* we need to keep track of this stuff on our own. the numpy array might get garbage collected */
  ierr = PetscMalloc1(msc->nmasks, &(ctx->masks));CHKERRQ(ierr);
  ierr = PetscMemcpy(ctx->masks, msc->masks, msc->nmasks*sizeof(PetscInt));CHKERRQ(ierr);

  ierr = PetscMalloc1(msc->nmasks+1, &(ctx->mask_offsets));CHKERRQ(ierr);
  ierr = PetscMemcpy(ctx->mask_offsets, msc->mask_offsets,
                     (msc->nmasks+1)*sizeof(PetscInt));CHKERRQ(ierr);

  ierr = PetscMalloc1(nterms, &(ctx->signs));CHKERRQ(ierr);
  ierr = PetscMemcpy(ctx->signs, msc->signs, nterms*sizeof(PetscInt));CHKERRQ(ierr);

  ierr = PetscMalloc1(nterms, &(ctx->real_coeffs));CHKERRQ(ierr);
  for (i=0; i < nterms; ++i) {
    real_part = PetscRealPart(msc->coeffs[i]);
    ctx->real_coeffs[i] = (real_part != 0) ? real_part : PetscImaginaryPart(msc->coeffs[i]);
  }

  ierr = C(CopySubspaceData,LEFT_SUBSPACE)(
    (C(data,LEFT_SUBSPACE)**)&(ctx->left_subspace_data),
    (C(data,LEFT_SUBSPACE)*)left_subspace_data);CHKERRQ(ierr);
  ierr = C(CopySubspaceData,RIGHT_SUBSPACE)(
    (C(data,RIGHT_SUBSPACE)**)&(ctx->right_subspace_data),
    (C(data,RIGHT_SUBSPACE)*)right_subspace_data);CHKERRQ(ierr);

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "MatDestroyCtx_CPU"
PetscErrorCode C(MatDestroyCtx_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A)
{
  PetscErrorCode ierr;
  shell_context *ctx;

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  ierr = PetscFree(ctx->masks);CHKERRQ(ierr);
  ierr = PetscFree(ctx->mask_offsets);CHKERRQ(ierr);
  ierr = PetscFree(ctx->signs);CHKERRQ(ierr);
  ierr = PetscFree(ctx->real_coeffs);CHKERRQ(ierr);

  ierr = C(DestroySubspaceData,LEFT_SUBSPACE)(ctx->left_subspace_data);CHKERRQ(ierr);
  ierr = C(DestroySubspaceData,RIGHT_SUBSPACE)(ctx->right_subspace_data);CHKERRQ(ierr);

  ierr = PetscFree(ctx);CHKERRQ(ierr);

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "MatMult_CPU"
/*
 * MatMult for CPU shell matrices.
 */
PetscErrorCode C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  PetscErrorCode ierr;
  int mpi_size, mpi_rank;
  MPI_Request send_request, recv_request;
  PetscInt x_size, x_local_size, max_proc_size, *x_local_sizes, *x_local_starts;
  PetscInt block_start, row_start, row_end, col_start, col_end;
  PetscInt proc_shift, proc_idx, send_count, recv_count;
  const PetscScalar* local_x_array;
  const PetscScalar* source_array;
  PetscScalar* x_array[2];
  PetscScalar* b_array;
  shell_context *ctx;

  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

  /* TODO: check that vectors are of correct type */

  ierr = MatShellGetContext(A, &ctx);CHKERRQ(ierr);

  ierr = VecSet(b, 0);CHKERRQ(ierr);

  ierr = VecGetArrayRead(x, &(local_x_array));CHKERRQ(ierr);
  ierr = VecGetArray(b, &(b_array));CHKERRQ(ierr);

  /* if there is only one process, just do one call to the kernel */
  if (mpi_size == 1) {

    ierr = VecGetOwnershipRange(x, &col_start, &col_end);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(b, &row_start, &row_end);CHKERRQ(ierr);

    C(MatMult_CPU_kernel,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
      local_x_array, b_array, ctx, row_start, row_end, col_start, col_end);CHKERRQ(ierr);
  }
  else {

    /* find the local sizes of all processors */
    ierr = VecGetSize(x, &x_size);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x, &x_local_size);CHKERRQ(ierr);
    ierr = PetscMalloc1(mpi_size, &x_local_sizes);CHKERRQ(ierr);
    ierr = MPI_Allgather(&x_local_size, 1, MPIU_INT, x_local_sizes, 1, MPIU_INT, PETSC_COMM_WORLD);

    ierr = PetscMalloc1(BLOCK_SIZE, &(x_array[0]));CHKERRQ(ierr);
    ierr = PetscMalloc1(BLOCK_SIZE, &(x_array[1]));CHKERRQ(ierr);

    /* compute the starting indices, and largest size on any processor */
    ierr = PetscMalloc1(mpi_size+1, &x_local_starts);CHKERRQ(ierr);
    max_proc_size = 0;
    x_local_starts[0] = 0;
    for (proc_idx = 0; proc_idx < mpi_size; ++proc_idx) {
      if (max_proc_size < x_local_sizes[proc_idx]) max_proc_size = x_local_sizes[proc_idx];
      x_local_starts[proc_idx+1] = x_local_starts[proc_idx] + x_local_sizes[proc_idx];
    }

    ierr = VecGetOwnershipRange(b, &row_start, &row_end);CHKERRQ(ierr);

    /* iterate through blocks on our process */
    for (block_start = 0; block_start < max_proc_size; block_start += BLOCK_SIZE) {

      for (proc_shift = 0; proc_shift < mpi_size; ++proc_shift) {
        /* do a round-robin of the data */
        /* eventually, skip the ones we don't need */
        proc_idx = (mpi_rank+proc_shift) % mpi_size;

        col_start = block_start + x_local_starts[proc_idx];
        col_end = PetscMin(col_start + BLOCK_SIZE, x_local_starts[proc_idx + 1]);

        /* self to self */
        if (proc_shift == 0) {
          source_array = local_x_array + block_start;
        }
        else {
          source_array = x_array[proc_shift%2];
          ierr = MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        }

        if (proc_shift < mpi_size - 1) {

          // TODO: put this stuff into a different function

          /* prepare to receive */
          recv_count = PetscMax(x_local_sizes[(mpi_rank+proc_shift+1)%mpi_size], block_start);
          recv_count = PetscMin(recv_count, BLOCK_SIZE + block_start) - block_start;
          if (recv_count > 0) {
            if (proc_shift > 0) {
              /* make sure we completed the previous send operation already */
              ierr = MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            }
            /* TODO: is this the right way to catch MPI errors with PETSc? */
            ierr = MPI_Irecv(x_array[(proc_shift+1)%2], recv_count, MPIU_SCALAR,
                              (mpi_rank+1)%mpi_size, 0,
                              PETSC_COMM_WORLD, &recv_request);
          }

          /* you guys silly I'm still gonna send it */
          send_count = col_end - col_start;
          if (send_count > 0) {
            /* TODO: is this the right way to catch MPI errors with PETSc? */
            ierr = MPI_Isend(source_array, send_count, MPIU_SCALAR,
                              (mpi_size+mpi_rank-1)%mpi_size, 0,
                              PETSC_COMM_WORLD, &send_request);
          }

        }

        /* finally actually do the computation */
        C(MatMult_CPU_kernel,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
          source_array, b_array, ctx, row_start, row_end, col_start, col_end);CHKERRQ(ierr);

      }
    }

    ierr = PetscFree(x_local_sizes);CHKERRQ(ierr);
    ierr = PetscFree(x_local_starts);CHKERRQ(ierr);
    ierr = PetscFree(x_array[0]);CHKERRQ(ierr);
    ierr = PetscFree(x_array[1]);CHKERRQ(ierr);

  }

  ierr = VecRestoreArrayRead(x, &(local_x_array));CHKERRQ(ierr);
  ierr = VecRestoreArray(b, (&b_array));CHKERRQ(ierr);

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "MatMult_CPU_kernel"
/*
 * MatMult kernel for CPU shell matrices.
 */
void C(MatMult_CPU_kernel,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const PetscScalar* x_array, PetscScalar* b_array, shell_context *ctx,
  PetscInt row_start, PetscInt row_end, PetscInt col_start, PetscInt col_end)
{
  PetscInt row_idx, ket, col_idx, bra;
  PetscInt mask_idx, term_idx;
  PetscInt sign;
  PetscScalar value;

  for (row_idx = row_start; row_idx < row_end; ++row_idx) {
    ket = C(I2S,LEFT_SUBSPACE)(row_idx, ctx->left_subspace_data);

    for (mask_idx = 0; mask_idx < ctx->nmasks; mask_idx++) {
      bra = ket ^ ctx->masks[mask_idx];
      col_idx = C(S2I,RIGHT_SUBSPACE)(bra, ctx->right_subspace_data);

      /* yikes */
      if (col_idx < col_start || col_idx >= col_end) continue;

      /* sum all terms for this matrix element */
      value = 0;
      for (term_idx = ctx->mask_offsets[mask_idx]; term_idx < ctx->mask_offsets[mask_idx+1]; ++term_idx) {
        sign = 1 - 2*(builtin_parity(bra & ctx->signs[term_idx]));
        if (TERM_REAL(ctx->masks[mask_idx], ctx->signs[term_idx])) {
          value += sign * ctx->real_coeffs[term_idx];
        } else {
          value += I * sign * ctx->real_coeffs[term_idx];
        }
      }
      b_array[row_idx - row_start] += value * x_array[col_idx - col_start];
    }
  }
}

#undef  __FUNCT__
#define __FUNCT__ "MatNorm_CPU"
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
        sign = 1 - 2*(builtin_parity(bra & ctx->signs[term_idx]));
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

  ierr = MPIU_Allreduce(&local_max, &global_max, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD);CHKERRQ(ierr);

  ctx->nrm = global_max;
  (*nrm) = global_max;

  return ierr;
}
