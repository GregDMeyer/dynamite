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
  PetscInt M, N, row_start, row_end, col_start;
  PetscInt mask_idx, term_idx, row_count;
  int mpi_size;
  PetscInt *diag_nonzeros, *offdiag_nonzeros;

  PetscInt row_idx, ket, col_idx, bra, sign;
  PetscScalar value;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));

  /* N is dimension of right subspace, M of left */
  M = C(Dim,LEFT_SUBSPACE)(left_subspace_data);
  N = C(Dim,RIGHT_SUBSPACE)(right_subspace_data);

  /* create matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetFromOptions(*A));

  /* TODO: we only should call these preallocation routines if matrix type is aij */
  /* preallocate memory */
  PetscCall(C(ComputeNonzeros,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))
	    (M, N, msc, &diag_nonzeros, &offdiag_nonzeros,
	     left_subspace_data, right_subspace_data));

  if (mpi_size == 1) {
    PetscCall(MatSeqAIJSetPreallocation(*A, 0, diag_nonzeros));
  }
  else {
    PetscCall(MatMPIAIJSetPreallocation(*A, 0, diag_nonzeros,
                                     0, offdiag_nonzeros));
  }

  /* this memory is allocated in ComputeNonzeros */
  PetscCall(PetscFree(diag_nonzeros));
  PetscCall(PetscFree(offdiag_nonzeros));

  /* compute matrix elements */
  PetscCall(MatSetOption(*A, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatGetOwnershipRange(*A, &row_start, &row_end));
  PetscCall(MatGetOwnershipRangeColumn(*A, &col_start, NULL));

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
      PetscCall(MatSetValue(*A, row_idx, col_idx, value, INSERT_VALUES));
    }

    /* workaround for a bug in PETSc that triggers if there are empty rows */
    if (row_count == 0) {
      PetscCall(MatSetValue(*A, row_idx, col_start, 0, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY));

  return 0;

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
  PetscInt mask_idx, row_idx, row_start, col_idx, col_start, state;
  PetscInt local_rows = PETSC_DECIDE;
  PetscInt local_cols = PETSC_DECIDE;
  PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &local_rows, &M));
  PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &local_cols, &N));

  /* prefix sum to get the start indices on each process */
  PetscCallMPI(MPI_Scan(&local_rows, &row_start, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Scan(&local_cols, &col_start, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  /* MPI_Scan includes current value in the sum */
  row_start -= local_rows;
  col_start -= local_cols;

  /* allocate storage for our diagonal and offdiagonal arrays */
  PetscCall(PetscCalloc1(local_rows, diag_nonzeros));
  PetscCall(PetscCalloc1(local_rows, offdiag_nonzeros));

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
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "BuildCPUShell"
PetscErrorCode C(BuildCPUShell,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const C(data,LEFT_SUBSPACE)* left_subspace_data,
  const C(data,RIGHT_SUBSPACE)* right_subspace_data,
  Mat *A)
{
  PetscInt M, N, m, n;
  shell_context *ctx;

  /* N is dimension of right subspace, M of left */
  M = C(Dim,LEFT_SUBSPACE)(left_subspace_data);
  N = C(Dim,RIGHT_SUBSPACE)(right_subspace_data);

  m = PETSC_DECIDE;
  n = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD,&m,&M);
  PetscSplitOwnership(PETSC_COMM_WORLD,&n,&N);

  PetscCall(C(BuildContext_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
    msc, left_subspace_data, right_subspace_data, &ctx));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, ctx, A));
  PetscCall(MatShellSetOperation(*A, MATOP_MULT,
				 (void(*)(void))C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))));
  PetscCall(MatShellSetOperation(*A, MATOP_NORM,
				 (void(*)(void))C(MatNorm_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))));
  PetscCall(MatShellSetOperation(*A, MATOP_DESTROY,
				 (void(*)(void))C(MatDestroyCtx_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))));

  return 0;
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
  shell_context *ctx;
  PetscInt nterms, i;
  PetscReal real_part;

  PetscCall(PetscMalloc1(1, ctx_p));
  ctx = (*ctx_p);

  ctx->nmasks = msc->nmasks;
  ctx->nrm = -1;
  nterms = msc->mask_offsets[msc->nmasks];

  /* we need to keep track of this stuff on our own. the numpy array might get garbage collected */
  PetscCall(PetscMalloc1(msc->nmasks, &(ctx->masks)));
  PetscCall(PetscMemcpy(ctx->masks, msc->masks, msc->nmasks*sizeof(PetscInt)));

  PetscCall(PetscMalloc1(msc->nmasks+1, &(ctx->mask_offsets)));
  PetscCall(PetscMemcpy(ctx->mask_offsets, msc->mask_offsets,
                     (msc->nmasks+1)*sizeof(PetscInt)));

  PetscCall(PetscMalloc1(nterms, &(ctx->signs)));
  PetscCall(PetscMemcpy(ctx->signs, msc->signs, nterms*sizeof(PetscInt)));

  PetscCall(PetscMalloc1(nterms, &(ctx->real_coeffs)));
  for (i=0; i < nterms; ++i) {
    real_part = PetscRealPart(msc->coeffs[i]);
    ctx->real_coeffs[i] = (real_part != 0) ? real_part : PetscImaginaryPart(msc->coeffs[i]);
  }

  PetscCall(C(CopySubspaceData,LEFT_SUBSPACE)(
    (C(data,LEFT_SUBSPACE)**)&(ctx->left_subspace_data),
    (C(data,LEFT_SUBSPACE)*)left_subspace_data));
  PetscCall(C(CopySubspaceData,RIGHT_SUBSPACE)(
    (C(data,RIGHT_SUBSPACE)**)&(ctx->right_subspace_data),
    (C(data,RIGHT_SUBSPACE)*)right_subspace_data));

  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "MatDestroyCtx_CPU"
PetscErrorCode C(MatDestroyCtx_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A)
{
  shell_context *ctx;

  PetscCall(MatShellGetContext(A,&ctx));

  PetscCall(PetscFree(ctx->masks));
  PetscCall(PetscFree(ctx->mask_offsets));
  PetscCall(PetscFree(ctx->signs));
  PetscCall(PetscFree(ctx->real_coeffs));

  PetscCall(C(DestroySubspaceData,LEFT_SUBSPACE)(ctx->left_subspace_data));
  PetscCall(C(DestroySubspaceData,RIGHT_SUBSPACE)(ctx->right_subspace_data));

  PetscCall(PetscFree(ctx));

  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "MatMult_CPU_General"

#undef VECSET_CACHE_SIZE
#ifdef PETSC_USE_DEBUG
  #define VECSET_CACHE_SIZE (PetscInt)(1<<7)
#else
  #define VECSET_CACHE_SIZE (PetscInt)(1<<15)
#endif

/*
 * MatMult for CPU shell matrices.
 */
PetscErrorCode C(MatMult_CPU_General,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  int mpi_size, mpi_rank;
  PetscBool assembling;

  PetscInt row_start, row_end, col_start, col_end, col_idx;
  PetscInt ket, bra, sign, row_idx, cache_idx, mask_idx, term_idx;

  PetscInt *row_idxs;
  PetscScalar value;
  const PetscScalar *local_x_array;
  PetscScalar *b_array, *to_send;

  int im_done, done_communicating;

  shell_context *ctx;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank));

  /* TODO: check that vectors are of correct type */

  PetscCall(MatShellGetContext(A, &ctx));

  PetscCall(VecSet(b, 0));

  PetscCall(VecGetArrayRead(x, &(local_x_array)));

  PetscCall(VecGetOwnershipRange(x, &col_start, &col_end));
  PetscCall(VecGetOwnershipRange(b, &row_start, &row_end));

  if (mpi_size == 1) {
    PetscCall(VecGetArray(b, &(b_array)));

    for (row_idx = row_start; row_idx < row_end; ++row_idx) {
      if (row_idx==row_start) {
        ket = C(I2S,LEFT_SUBSPACE)(row_idx, ctx->left_subspace_data);
      } else {
        ket = C(NextState,LEFT_SUBSPACE)(ket, row_idx, ctx->left_subspace_data);
      }

      for (mask_idx = 0; mask_idx < ctx->nmasks; mask_idx++) {
        bra = ket ^ ctx->masks[mask_idx];

        col_idx = C(S2I,RIGHT_SUBSPACE)(bra, ctx->right_subspace_data);

        if (col_idx == -1) continue;

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
        b_array[row_idx - row_start] += value * local_x_array[col_idx - col_start];
      }
    }

    PetscCall(VecRestoreArray(b, &b_array));
  }
  else {

    /* allocate for cache */
    PetscCall(PetscMalloc1(VECSET_CACHE_SIZE, &row_idxs));
    PetscCall(PetscMalloc1(VECSET_CACHE_SIZE, &to_send));

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

  	row_idx = C(S2I,LEFT_SUBSPACE)(bra, ctx->left_subspace_data);

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
	  SETERRQ(MPI_COMM_SELF, PETSC_ERR_MEMC, "cache out of bounds, value %d", cache_idx);
	}

	row_idxs[cache_idx] = row_idx;
	to_send[cache_idx] = value * local_x_array[col_idx-col_start];
	++cache_idx;

	if (cache_idx == VECSET_CACHE_SIZE) {
	  if (assembling) {
	    PetscCall(VecAssemblyEnd(b));
	    im_done = 0;
	    PetscCallMPI(MPI_Allreduce(&im_done, &done_communicating, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD));
	    assembling = PETSC_FALSE;
	  }

	  PetscCall(VecSetValues(b, VECSET_CACHE_SIZE, row_idxs, to_send, ADD_VALUES));

	  PetscCall(VecAssemblyBegin(b));
	  assembling = PETSC_TRUE;

	  cache_idx = 0;
	}
      }
    }

    im_done = cache_idx==0;
    done_communicating = PETSC_FALSE;
    if (assembling) {
      PetscCall(VecAssemblyEnd(b));
      PetscCallMPI(MPI_Allreduce(&im_done, &done_communicating, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD));
    }

    if (!im_done) {
      PetscCall(VecSetValues(b, cache_idx, row_idxs, to_send, ADD_VALUES));
    }

    while (!done_communicating) {
      PetscCall(VecAssemblyBegin(b));
      PetscCall(VecAssemblyEnd(b));
      im_done = 1;
      PetscCallMPI(MPI_Allreduce(&im_done, &done_communicating, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD));
    }

    PetscCall(PetscFree(row_idxs));
    PetscCall(PetscFree(to_send));
  }

  PetscCall(VecRestoreArrayRead(x, &local_x_array));

  return 0;
}

/* use the hand-tuned kernel for parity and full subspaces, if we can */
/* if subspaces are the same, and are both Full or Parity, use the fancy fast matvec */
#if C(LEFT_SUBSPACE,SP) == C(RIGHT_SUBSPACE,SP) && (C(LEFT_SUBSPACE,SP) == Full_SP || C(LEFT_SUBSPACE,SP) == Parity_SP)

#define ITER_CUTOFF (PetscInt)8
#define LKP_MASK (LKP_SIZE-1)

#undef VECSET_CACHE_SIZE

#ifdef PETSC_USE_DEBUG
  #define VECSET_CACHE_SIZE (PetscInt)(1<<7)
  #define LKP_SIZE (PetscInt)(1<<3)
#else
  #define VECSET_CACHE_SIZE (PetscInt)(1<<11)
  #define LKP_SIZE (PetscInt)(1<<6)
#endif

PetscErrorCode C(MatMult_CPU_Fast,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b);

PetscErrorCode C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  PetscBool use_fast_matmult;
  PetscInt local_size;
  shell_context *ctx;
  int mpi_size;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&mpi_size));

  PetscCall(VecGetLocalSize(b, &local_size));

  PetscCall(MatShellGetContext(A,&ctx));

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
    PetscCall(C(MatMult_CPU_Fast,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(A, x, b));
  }
  else {
    PetscCall(C(MatMult_CPU_General,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(A, x, b));
  }
  return 0;
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
  PetscInt iterate_max, cache_idx, inner_idx, row_idx, local_start, stop;

  iterate_max = (PetscInt)1 << builtin_ctz(mask);
  PetscAssert(iterate_max > (PetscInt)0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "iterate_max %ld <= 0", iterate_max);

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
      local_start = row_idx-x_start;
      PetscAssert(local_start >= (PetscInt)0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "negative index %ld on x array", local_start);
      PetscAssert(local_start+stop-(PetscInt)1 < x_end-x_start, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index %ld past end %ld of x array", local_start+stop-(PetscInt)1, x_end-x_start);
      for (inner_idx=0; inner_idx < stop; ++inner_idx) {
        values[cache_idx+inner_idx] += summed_c[cache_idx+inner_idx] * x_array[local_start+inner_idx];
      }
      cache_idx += inner_idx;
    }
  }

  return 0;

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
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&mpi_rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&mpi_size));

  /* check if number of processors is a multiple of 2 */
  if ((mpi_size & (mpi_size-1)) != 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "number of MPI procs must be a power of 2");
  }

  PetscCall(MatShellGetContext(A,&ctx));

  /* clear out the b vector */
  PetscCall(VecSet(b,0));

  /* prepare x array */
  PetscCall(VecGetOwnershipRange(x, &x_start, &x_end));
  PetscCall(VecGetArrayRead(x, &x_array));

  /* allocate for cache */
  PetscCall(PetscMalloc1(VECSET_CACHE_SIZE, &row_idx));
  PetscCall(PetscMalloc1(VECSET_CACHE_SIZE, &summed_coeffs));
  PetscCall(PetscMalloc1(VECSET_CACHE_SIZE, &values));

  PetscCall(PetscMalloc1(LKP_SIZE*LKP_SIZE, &lookup));
  compute_sign_lookup(lookup);

  #if (C(LEFT_SUBSPACE,SP) == Parity_SP)
    PetscCall(PetscMalloc1(LKP_SIZE*LKP_SIZE, &parity_lookup));
    compute_parity_sign_lookup(((data_Parity*)(ctx->left_subspace_data))->space, parity_lookup);
  #endif

  /* this relies on MPI size being a power of 2 */
  /* this is log base 2 of the local vector size */
  n_local_spins = builtin_ctz(x_end - x_start);

  PetscCall(PetscMalloc1(mpi_size+1,&(mask_starts)));
  C(compute_mask_starts,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
    ctx->nmasks,
    n_local_spins,
    mpi_size,
    ctx->masks,
    mask_starts
  );

  proc_mask = (PetscInt)(-1) << n_local_spins;
  proc_me = (PetscInt)mpi_rank << n_local_spins;

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

      PetscCall(PetscMemzero(values,   sizeof(PetscScalar)*VECSET_CACHE_SIZE));
      PetscCall(PetscMemzero(summed_coeffs, sizeof(PetscScalar)*VECSET_CACHE_SIZE));

      for (cache_idx=0; cache_idx < VECSET_CACHE_SIZE; ++cache_idx) {
        row_idx[cache_idx] = block_start_idx+cache_idx;
      }

      for (mask_idx = mask_starts[proc_idx]; mask_idx < mask_starts[proc_idx+1]; ++mask_idx) {

        #if (C(LEFT_SUBSPACE,SP) == Parity_SP)
        /* skip terms that don't preserve parity */
        if (builtin_parity(ctx->masks[mask_idx])) {
          continue;
        }
        #endif

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

        PetscCall(do_cache_product(m, block_start_idx, x_start, x_end, summed_coeffs, x_array, values));
        PetscCall(PetscMemzero(summed_coeffs, sizeof(PetscScalar)*VECSET_CACHE_SIZE));

      }

      if (assembling) {
        PetscCall(VecAssemblyEnd(b));
        assembling = PETSC_FALSE;
      }
      PetscCall(VecSetValues(b, VECSET_CACHE_SIZE, row_idx, values, ADD_VALUES));

      PetscCall(VecAssemblyBegin(b));
      assembling = PETSC_TRUE;
    }
  }

  if (assembling) {
    PetscCall(VecAssemblyEnd(b));
  }

  PetscCall(VecRestoreArrayRead(x,&x_array));

  PetscCall(PetscFree(lookup));
  PetscCall(PetscFree(row_idx));
  PetscCall(PetscFree(values));
  PetscCall(PetscFree(summed_coeffs));

  return 0;
}

#else

PetscErrorCode C(MatMult_CPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  PetscCall(C(MatMult_CPU_General,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(A,x,b));
  return 0;
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
  PetscInt row_idx, row_start, row_end;
  PetscInt mask_idx, term_idx, ket, bra;
  PetscInt sign;
  PetscScalar csum;
  PetscReal sum, sum_err, comp, total, local_max, global_max;
  shell_context *ctx;

  if (type != NORM_INFINITY) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Only NORM_INFINITY is implemented for shell matrices.");
  }

  PetscCall(MatShellGetContext(A, &ctx));

  /*
   * keep the norm cached so we don't have to compute it all the time.
   * if we already have it, just return it
   */
  if (ctx->nrm != -1) {
    (*nrm) = ctx->nrm;
    return 0;
  }

  PetscCall(MatGetOwnershipRange(A, &row_start, &row_end));

  local_max = 0;
  for (row_idx = row_start; row_idx < row_end; ++row_idx) {

    ket = C(I2S,LEFT_SUBSPACE)(row_idx, ctx->left_subspace_data);

    /* sum abs of all matrix elements in this row */
    /* for precision reasons use the Kahan summation algorithm */
    sum = 0;
    sum_err = 0;
    for (mask_idx = 0; mask_idx < ctx->nmasks; ++mask_idx) {

      bra = ket ^ ctx->masks[mask_idx];

      if (C(S2I,RIGHT_SUBSPACE)(bra, ctx->right_subspace_data) == -1) {
	continue;
      }

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

  PetscCallMPI(MPIU_Allreduce(&local_max, &global_max, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));

  ctx->nrm = global_max;
  (*nrm) = global_max;

  return 0;
}

/*
 * This function checks whether an operator (represented as MSC) conserves the given
 * subspaces. It assumes a product state basis; extra conservation laws (such as the
 * extra Z2 symmetry in SpinConserve+spinflip) need to be checked externally.
 */
#undef  __FUNCT__
#define __FUNCT__ "CheckConserves"
PetscErrorCode C(CheckConserves,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const C(data,LEFT_SUBSPACE)* left_subspace_data,
  const C(data,RIGHT_SUBSPACE)* right_subspace_data,
  PetscInt* result)
{
  PetscLayout layout;
  PetscInt N, col_start, col_end;
  PetscInt mask_idx, term_idx;
  PetscInt row_idx, ket, col_idx, bra, sign;
  PetscScalar value;
  PetscInt local_result;

  /* dimension of right subspace */
  N = C(Dim,RIGHT_SUBSPACE)(right_subspace_data);

  /* split the work across processes */
  PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layout));
  PetscCall(PetscLayoutSetSize(layout, N));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetRange(layout, &col_start, &col_end));

  local_result = 1;

  for (col_idx=col_start; col_idx<col_end; ++col_idx) {

    /* each term looks like value*|ket><bra| */
    bra = C(I2S,RIGHT_SUBSPACE)(col_idx, right_subspace_data);

    for (mask_idx=0; mask_idx<msc->nmasks; mask_idx++) {
      ket = bra ^ msc->masks[mask_idx];

      row_idx = C(S2I,LEFT_SUBSPACE)(ket, left_subspace_data);

      /* in this case, it mapped onto a row that was in the subspace, so we're good */
      if (row_idx != -1) {
        continue;
      }

      /* otherwise, if the sum of all terms for this matrix element is 0, we're OK */
      value = 0;
      for (term_idx=msc->mask_offsets[mask_idx]; term_idx<msc->mask_offsets[mask_idx+1]; ++term_idx) {
        sign = 1 - 2*(builtin_parity(bra & msc->signs[term_idx]));
        value += sign * msc->coeffs[term_idx];
      }

      /* if value == 0, we can just forget about it and continue */
      if (value != 0) {
	local_result = 0;
	break;
      }
    }
    if (!local_result) break;
  }

  /* communicate the result among everybody */
  PetscCallMPI(MPI_Allreduce(&local_result, result, 1, MPIU_INT, MPI_LAND, PETSC_COMM_WORLD));

  /* global result is now stored in result, we're done */

  return 0;
}
