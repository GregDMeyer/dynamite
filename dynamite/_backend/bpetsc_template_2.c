/*
 * This file defines a template for matrix functions. It should be included
 * multiple times in bpetsc_impl.c, with LEFT_SUBSPACE and RIGHT_SUBSPACE defined as
 * the desired values.
 */

#include "bpetsc_template_2.h"
#if PETSC_HAVE_CUDA
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
#if PETSC_HAVE_CUDA
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

#if C(RIGHT_SUBSPACE,SP) == SpinConserve_SP
  PetscInt s2i_sign;
#endif

  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);CHKERRMPI(ierr);

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

#if C(RIGHT_SUBSPACE,SP) == SpinConserve_SP
      col_idx = C(S2I,RIGHT_SUBSPACE)(bra, &s2i_sign, right_subspace_data);
#else
      col_idx = C(S2I,RIGHT_SUBSPACE)(bra, right_subspace_data);
#endif
      if (col_idx == -1) {
        continue;
      }

      /* sum all terms for this matrix element */
      value = 0;
      for (term_idx = msc->mask_offsets[mask_idx]; term_idx < msc->mask_offsets[mask_idx+1]; ++term_idx) {
        sign = 1 - 2*(builtin_parity(bra & msc->signs[term_idx]));
        value += sign * msc->coeffs[term_idx];
      }

#if C(RIGHT_SUBSPACE,SP) == SpinConserve_SP
      value *= s2i_sign;
#endif

      row_count++;
      ierr = MatSetValue(*A, row_idx, col_idx, value, ADD_VALUES);CHKERRQ(ierr);
    }

    /* workaround for a bug in PETSc that triggers if there are empty rows */
    if (row_count == 0) {
      ierr = MatSetValue(*A, row_idx, col_start, 0, ADD_VALUES);CHKERRQ(ierr);
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
  ierr = MPI_Scan(&local_rows, &row_start, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = MPI_Scan(&local_cols, &col_start, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);CHKERRMPI(ierr);

  /* MPI_Scan includes current value in the sum */
  row_start -= local_rows;
  col_start -= local_cols;

  /* allocate storage for our diagonal and offdiagonal arrays */
  ierr = PetscCalloc1(local_rows, diag_nonzeros);CHKERRQ(ierr);
  ierr = PetscCalloc1(local_rows, offdiag_nonzeros);CHKERRQ(ierr);

  for (row_idx = 0; row_idx < local_rows; row_idx++) {
    state = C(I2S,LEFT_SUBSPACE)(row_idx+row_start, left_subspace_data);
    for (mask_idx = 0; mask_idx < msc->nmasks; ++mask_idx) {

#if C(RIGHT_SUBSPACE,SP) == SpinConserve_SP
      col_idx = C(S2I,RIGHT_SUBSPACE)(state^msc->masks[mask_idx], NULL, right_subspace_data);
#else
      col_idx = C(S2I,RIGHT_SUBSPACE)(state^msc->masks[mask_idx], right_subspace_data);
#endif

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
#define __FUNCT__ "MatMult_CPU_General"

#undef VECSET_CACHE_SIZE
#ifdef PETSC_USE_DEBUG
  #define VECSET_CACHE_SIZE (1<<7)
#else
  #define VECSET_CACHE_SIZE (1<<15)
#endif

/*
 * MatMult for CPU shell matrices.
 */
PetscErrorCode C(MatMult_CPU_General,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  PetscErrorCode ierr;
  int mpi_size, mpi_rank;
  PetscBool assembling;

  PetscInt row_start, row_end, col_start, col_end, col_idx;
  PetscInt ket, bra, sign, row_idx, cache_idx, mask_idx, term_idx;

#if C(LEFT_SUBSPACE,SP) == SpinConserve_SP
  PetscInt s2i_sign;
#endif

  PetscInt *row_idxs;
  PetscScalar value;
  const PetscScalar *local_x_array;
  PetscScalar *b_array, *to_send;

  int im_done, done_communicating;

  shell_context *ctx;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);CHKERRMPI(ierr);

  /* TODO: check that vectors are of correct type */

  ierr = MatShellGetContext(A, &ctx);CHKERRQ(ierr);

  ierr = VecSet(b, 0);CHKERRQ(ierr);

  ierr = VecGetArrayRead(x, &(local_x_array));CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x, &col_start, &col_end);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(b, &row_start, &row_end);CHKERRQ(ierr);

  /* if there is only one process, just do one call to the kernel */
  if (mpi_size == 1) {
    ierr = VecGetArray(b, &(b_array));CHKERRQ(ierr);

    C(MatMult_CPU_kernel,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
      local_x_array, b_array, ctx, row_start, row_end, col_start, col_end);CHKERRQ(ierr);

    ierr = VecRestoreArray(b, &b_array);CHKERRQ(ierr);
  }
  else {

    /* allocate for cache */
    ierr = PetscMalloc1(VECSET_CACHE_SIZE, &row_idxs);CHKERRQ(ierr);
    ierr = PetscMalloc1(VECSET_CACHE_SIZE, &to_send);CHKERRQ(ierr);

    /* we are not already sending values to another processor */
    assembling = PETSC_FALSE;

    cache_idx = 0;

    ket = 0;
    for (col_idx=col_start; col_idx<col_end; ++col_idx) {

      if (col_idx==col_start) {
	ket = C(I2S,RIGHT_SUBSPACE)(col_idx, ctx->right_subspace_data);
      } else {
	ket = C(NextState,RIGHT_SUBSPACE)(ket, col_idx, ctx->right_subspace_data);
      }

      for (mask_idx=0; mask_idx<ctx->nmasks; mask_idx++) {
	bra = ket ^ ctx->masks[mask_idx];

#if C(LEFT_SUBSPACE,SP) == SpinConserve_SP
	row_idx = C(S2I,LEFT_SUBSPACE)(bra, &s2i_sign, ctx->left_subspace_data);
#else
  	row_idx = C(S2I,LEFT_SUBSPACE)(bra, ctx->left_subspace_data);
#endif

	if (row_idx == -1) continue;

	/* sum all terms for this matrix element */
	value = 0;
	for (term_idx=ctx->mask_offsets[mask_idx]; term_idx<ctx->mask_offsets[mask_idx+1]; ++term_idx) {
	  sign = 1 - 2*(builtin_parity(ket&ctx->signs[term_idx]));
	  if (TERM_REAL(ctx->masks[mask_idx], ctx->signs[term_idx])) {
	    value += sign * ctx->real_coeffs[term_idx];
	  } else {
	    value += I * sign * ctx->real_coeffs[term_idx];
	  }
	}
	if (cache_idx >= VECSET_CACHE_SIZE) {
	  SETERRQ1(MPI_COMM_SELF, PETSC_ERR_MEMC, "cache out of bounds, value %d", cache_idx);
	}

#if C(LEFT_SUBSPACE,SP) == SpinConserve_SP
	value *= s2i_sign;
#endif

	row_idxs[cache_idx] = row_idx;
	to_send[cache_idx] = value * local_x_array[col_idx-col_start];
	++cache_idx;

	if (cache_idx == VECSET_CACHE_SIZE) {
	  if (assembling) {
	    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
	    im_done = 0;
	    ierr = MPI_Allreduce(&im_done, &done_communicating, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);CHKERRMPI(ierr);
	    assembling = PETSC_FALSE;
	  }

	  ierr = VecSetValues(b, VECSET_CACHE_SIZE, row_idxs, to_send, ADD_VALUES);CHKERRQ(ierr);

	  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
	  assembling = PETSC_TRUE;

	  cache_idx = 0;
	}
      }
    }

    im_done = cache_idx==0;
    done_communicating = PETSC_FALSE;
    if (assembling) {
      ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&im_done, &done_communicating, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);CHKERRMPI(ierr);
    }

    if (!im_done) {
      ierr = VecSetValues(b, cache_idx, row_idxs, to_send, ADD_VALUES);CHKERRQ(ierr);
    }

    while (!done_communicating) {
      ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
      im_done = 1;
      ierr = MPI_Allreduce(&im_done, &done_communicating, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);CHKERRMPI(ierr);
    }

    ierr = PetscFree(row_idxs);CHKERRQ(ierr);
    ierr = PetscFree(to_send);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(x, &local_x_array);CHKERRQ(ierr);

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
#if C(RIGHT_SUBSPACE,SP) == SpinConserve_SP
  PetscInt s2i_sign=0;
#endif
  PetscScalar value;

  for (row_idx = row_start; row_idx < row_end; ++row_idx) {
    ket = C(I2S,LEFT_SUBSPACE)(row_idx, ctx->left_subspace_data);

    for (mask_idx = 0; mask_idx < ctx->nmasks; mask_idx++) {
      bra = ket ^ ctx->masks[mask_idx];

#if C(RIGHT_SUBSPACE,SP) == SpinConserve_SP
      col_idx = C(S2I,RIGHT_SUBSPACE)(bra, &s2i_sign, ctx->right_subspace_data);
#else
      col_idx = C(S2I,RIGHT_SUBSPACE)(bra, ctx->right_subspace_data);
#endif

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
#if C(RIGHT_SUBSPACE,SP) == SpinConserve_SP
      value *= s2i_sign;
#endif
      b_array[row_idx - row_start] += value * x_array[col_idx - col_start];
    }
  }
}

/* use the hand-tuned kernel for parity and full subspaces, if we can */
/* if subspaces are the same, and are both Full or Parity, use the fancy fast matvec */
#if C(LEFT_SUBSPACE,SP) == C(RIGHT_SUBSPACE,SP) && (C(LEFT_SUBSPACE,SP) == Full_SP || C(LEFT_SUBSPACE,SP) == Parity_SP)

#define ITER_CUTOFF 8
#define LKP_MASK (LKP_SIZE-1)

#undef VECSET_CACHE_SIZE

#ifdef PETSC_USE_DEBUG
  #define VECSET_CACHE_SIZE (1<<7)
  #define LKP_SIZE (1<<3)
#else
  #define VECSET_CACHE_SIZE (1<<11)
  #define LKP_SIZE (1<<6)
#endif

PetscErrorCode C(MatMult_CPU_Fast,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b);

PetscErrorCode C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  PetscErrorCode ierr;
  PetscBool use_fast_matmult;
  PetscInt local_size;
  shell_context *ctx;
  int mpi_size;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpi_size);CHKERRMPI(ierr);

  ierr = VecGetLocalSize(b, &local_size);CHKERRQ(ierr);

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  /* mpi size is a power of 2 */
  use_fast_matmult = __builtin_popcount(mpi_size) == 1;

  /* problem size is big enough */
  use_fast_matmult = use_fast_matmult && (local_size > VECSET_CACHE_SIZE);

  #if (C(LEFT_SUBSPACE,SP) == Parity_SP)
    use_fast_matmult = use_fast_matmult && (\
      ((data_Parity*)(ctx->left_subspace_data))->space == \
      ((data_Parity*)(ctx->right_subspace_data))->space);
  #endif

  if (use_fast_matmult) {
    ierr = C(MatMult_CPU_Fast,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(A, x, b);CHKERRQ(ierr);
  }
  else {
    ierr = C(MatMult_CPU_General,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(A, x, b);CHKERRQ(ierr);
  }
  return ierr;
}

#ifndef HELPER_FNS
#define HELPER_FNS 1
static inline void _radd(PetscScalar *x,PetscReal c)
{
  (*x) += c;
}

static inline void _cadd(PetscScalar *x,PetscReal c)
{
  (*x) += I*c;
}

void compute_sign_lookup(PetscInt* lookup)
{
  PetscInt i, j, tmp;
  for (i=0;i<LKP_SIZE;++i) {
    for (j=0;j<LKP_SIZE;++j) {
      tmp = builtin_parity(i&j);
      lookup[i*LKP_SIZE + j] = -(tmp^(tmp-1));
    }
  }
}

void compute_parity_sign_lookup(PetscInt parity, PetscInt* lookup)
{
  PetscInt i, j, tmp;
  for (i=0;i<LKP_SIZE;++i) {
    for (j=0;j<LKP_SIZE;++j) {
      tmp = builtin_parity(i&j);
      tmp ^= builtin_parity(j) ^ parity;
      lookup[i*LKP_SIZE + j] = -(tmp^(tmp-1));
    }
  }
}

PetscErrorCode do_cache_product(
  PetscInt mask,
  PetscInt block_start,
  PetscInt x_start,
  PetscInt x_end,
  const PetscScalar* summed_c,
  const PetscScalar* x_array,
  PetscScalar* values
)
{
  PetscInt iterate_max, cache_idx, inner_idx, row_idx, stop;
  PetscErrorCode ierr = 0;

  /* TODO: this is not compatible with 64 bit ints */
  iterate_max = 1 << __builtin_ctz(mask);
  if (iterate_max < ITER_CUTOFF) {
    for (cache_idx=0; cache_idx < VECSET_CACHE_SIZE; ++cache_idx) {
      row_idx = (block_start+cache_idx) ^ mask;
      values[cache_idx] += summed_c[cache_idx]*x_array[row_idx-x_start];
    }
  }
  else {
    for (cache_idx=0; cache_idx < VECSET_CACHE_SIZE;) {
      row_idx = (block_start+cache_idx) ^ mask;
      stop = intmin(iterate_max-(row_idx%iterate_max), VECSET_CACHE_SIZE-cache_idx);
      for (inner_idx=0; inner_idx < stop; ++inner_idx) {
        if (row_idx+inner_idx-x_start < 0) {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEMC, "negative index on x array");
        }
        if (row_idx+inner_idx-x_start >= x_end-x_start) {
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEMC, "index past end of x array");
        }
        values[cache_idx+inner_idx] += summed_c[cache_idx+inner_idx] * x_array[row_idx+inner_idx-x_start];
      }
      cache_idx += inner_idx;
    }
  }

  return ierr;

}

void sum_term(
  PetscInt block_start,
  PetscInt sign,
  PetscInt is_real,
  PetscScalar coeff,
  PetscBool check_parity,
  const PetscInt* lookup,
  PetscScalar* summed_c
) {
  PetscInt cache_idx, lkp_idx, flip;
  PetscScalar tmp_c;

  const PetscInt* l;
  l = lookup + (sign&LKP_MASK)*LKP_SIZE;

/* this is the interior of the for loop. The compiler wasn't
 * doing a good enough job unswitching it so I write a macro
 * to unswitch it manually.
 */
/* TODO: include sign flips due to parity bit in lookup table */
/**********/
#define INNER_LOOP(sign_flip,add_func,parity_check)                     \
  for (cache_idx=0; cache_idx<VECSET_CACHE_SIZE; ) {                    \
    flip = builtin_parity((cache_idx+block_start)&(~LKP_MASK)&sign);    \
    if (parity_check)                                                   \
      flip ^= builtin_parity((cache_idx+block_start)&(~LKP_MASK));      \
    tmp_c = -(flip^(flip-1))*coeff;                                     \
    for (lkp_idx=0; lkp_idx<LKP_SIZE; ++lkp_idx,++cache_idx) {          \
      add_func(summed_c+cache_idx,(sign_flip)*tmp_c);                   \
    }                                                                   \
  }
/**********/

  if (check_parity) {
    if (is_real) {INNER_LOOP(l[lkp_idx],_radd,1)}
    else {INNER_LOOP(l[lkp_idx],_cadd,1)}
  }
  else {
    if (sign&LKP_MASK) {
      if (is_real) {INNER_LOOP(l[lkp_idx],_radd,0)}
      else {INNER_LOOP(l[lkp_idx],_cadd,0)}
    }
    else {
      if (is_real) {INNER_LOOP(1,_radd,0)}
      else {INNER_LOOP(1,_cadd,0)}
    }
  }
}
#endif

void C(compute_mask_starts,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  PetscInt nmasks,
  PetscInt n_local_spins,
  PetscInt mpi_size,
  const PetscInt* masks,
  PetscInt* mask_starts
)
{
  PetscInt proc_idx, mask_idx;

  mask_idx = 0;
  for (proc_idx = 0; proc_idx < mpi_size; ++proc_idx) {
    // search for the first mask that has the same prefix as this process
    // in parity case, we drop the last bit of the mask (by calling S2I on it)
    while (
        mask_idx < nmasks &&
        C(S2I_nocheck,LEFT_SUBSPACE)(masks[mask_idx], NULL) < (proc_idx << n_local_spins)
      ) {
      ++mask_idx;
    }
    mask_starts[proc_idx] = mask_idx;
  }
  mask_starts[mpi_size] = nmasks;
}

#undef  __FUNCT__
#define __FUNCT__ "MatMult_CPU_Fast"
PetscErrorCode C(MatMult_CPU_Fast,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  PetscErrorCode ierr;
  PetscInt proc_idx, proc_me, proc_mask, n_local_spins;
  PetscInt proc_start_idx, block_start_idx;
  PetscInt mask_idx, term_idx;
  PetscInt m, s, ms_parity;
  PetscReal c;
  PetscInt *mask_starts, *lookup;
  PetscBool assembling, r;

  #if (C(LEFT_SUBSPACE,SP) == Parity_SP)
  PetscInt *parity_lookup;
  #endif

  PetscInt x_start, x_end;
  const PetscScalar *x_array;

  shell_context *ctx;

  /* cache */
  PetscInt *row_idx;
  PetscScalar *summed_coeffs, *values;
  PetscInt cache_idx;

  int mpi_rank,mpi_size;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&mpi_rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpi_size);CHKERRMPI(ierr);

  /* check if number of processors is a multiple of 2 */
  if ((mpi_size & (mpi_size-1)) != 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "number of MPI procs must be a power of 2");
  }

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  /* clear out the b vector */
  ierr = VecSet(b,0);CHKERRQ(ierr);

  /* prepare x array */
  ierr = VecGetOwnershipRange(x, &x_start, &x_end);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &x_array);CHKERRQ(ierr);

  /* allocate for cache */
  ierr = PetscMalloc1(VECSET_CACHE_SIZE, &row_idx);CHKERRQ(ierr);
  ierr = PetscMalloc1(VECSET_CACHE_SIZE, &summed_coeffs);CHKERRQ(ierr);
  ierr = PetscMalloc1(VECSET_CACHE_SIZE, &values);CHKERRQ(ierr);

  ierr = PetscMalloc1(LKP_SIZE*LKP_SIZE, &lookup);CHKERRQ(ierr);
  compute_sign_lookup(lookup);

  #if (C(LEFT_SUBSPACE,SP) == Parity_SP)
    ierr = PetscMalloc1(LKP_SIZE*LKP_SIZE, &parity_lookup);CHKERRQ(ierr);
    compute_parity_sign_lookup(((data_Parity*)(ctx->left_subspace_data))->space, parity_lookup);
  #endif

  /* this relies on MPI size being a power of 2 */
  /* this is log base 2 of the local vector size */
  n_local_spins = __builtin_ctz(x_end - x_start);

  ierr = PetscMalloc1(mpi_size+1,&(mask_starts));CHKERRQ(ierr);
  C(compute_mask_starts,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
    ctx->nmasks,
    n_local_spins,
    mpi_size,
    ctx->masks,
    mask_starts
  );

  proc_mask = (-1) << n_local_spins;
  proc_me = mpi_rank << n_local_spins;

  /* we are not already sending values to another processor */
  assembling = PETSC_FALSE;

  for (proc_idx = 0; proc_idx < mpi_size; ++proc_idx) {

    /* if there are none for this process, skip it */
    if (mask_starts[proc_idx] == mask_starts[proc_idx+1]) continue;

    /* if we've hit the end of the masks, stop */
    if (mask_starts[proc_idx] == ctx->nmasks) break;

    /* the first index of the target */
    m = C(S2I_nocheck,LEFT_SUBSPACE)(ctx->masks[mask_starts[proc_idx]], NULL);
    proc_start_idx = proc_mask & (proc_me ^ m);

    for (block_start_idx = proc_start_idx;
         block_start_idx < proc_start_idx + (x_end - x_start);
         block_start_idx += VECSET_CACHE_SIZE) {

      ierr = PetscMemzero(values,   sizeof(PetscScalar)*VECSET_CACHE_SIZE);CHKERRQ(ierr);
      ierr = PetscMemzero(summed_coeffs, sizeof(PetscScalar)*VECSET_CACHE_SIZE);CHKERRQ(ierr);

      for (cache_idx=0; cache_idx < VECSET_CACHE_SIZE; ++cache_idx) {
        row_idx[cache_idx] = block_start_idx+cache_idx;
      }

      for (mask_idx = mask_starts[proc_idx]; mask_idx < mask_starts[proc_idx+1]; ++mask_idx) {

        m = C(S2I_nocheck,LEFT_SUBSPACE)(
          ctx->masks[mask_idx],
          NULL
        );

        for (
            term_idx = ctx->mask_offsets[mask_idx];
            term_idx < ctx->mask_offsets[mask_idx+1];
            ++term_idx) {

          s = C(S2I_nocheck,LEFT_SUBSPACE)(
            ctx->signs[term_idx],
            NULL
          );

          ms_parity = builtin_parity(ctx->masks[mask_idx] & ctx->signs[term_idx]);
          c = -(ms_parity^(ms_parity-1))*ctx->real_coeffs[term_idx];
          r = TERM_REAL(ctx->masks[mask_idx], ctx->signs[term_idx]);

          #if (C(LEFT_SUBSPACE,SP) == Parity_SP)
            if (ctx->signs[term_idx] & 1) {
              sum_term(block_start_idx, s, r, c, 1, parity_lookup, summed_coeffs);
            }
            else {
              sum_term(block_start_idx, s, r, c, 0, lookup, summed_coeffs);
            }
          #else
            sum_term(block_start_idx, s, r, c, 0, lookup, summed_coeffs);
          #endif

        }

        ierr = do_cache_product(m, block_start_idx, x_start, x_end, summed_coeffs, x_array, values);CHKERRQ(ierr);
        ierr = PetscMemzero(summed_coeffs, sizeof(PetscScalar)*VECSET_CACHE_SIZE);CHKERRQ(ierr);

      }

      if (assembling) {
        ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
        assembling = PETSC_FALSE;
      }
      ierr = VecSetValues(b, VECSET_CACHE_SIZE, row_idx, values, ADD_VALUES);CHKERRQ(ierr);

      ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
      assembling = PETSC_TRUE;
    }
  }

  if (assembling) {
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);

  ierr = PetscFree(lookup);CHKERRQ(ierr);
  ierr = PetscFree(row_idx);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(summed_coeffs);CHKERRQ(ierr);

  return ierr;
}

#else

PetscErrorCode C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  PetscErrorCode ierr;
  ierr = C(MatMult_CPU_General,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(A,x,b);CHKERRQ(ierr);
  return ierr;
}

#endif

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

      // extra s2i sign of csum doesn't matter because we are
      // immediately taking the absolute value
      // TODO: handle the extreme edge case in which two different terms collide
      // onto the same matrix element.
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
