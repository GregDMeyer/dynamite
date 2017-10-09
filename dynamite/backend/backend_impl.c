
#include "backend_impl.h"

#undef  __FUNCT__
#define __FUNCT__ "BuildMat_Full"
PetscErrorCode BuildMat_Full(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt N,i,state,Istart,Iend,nrows,local_bits,nonlocal_mask;
  int mpi_rank,mpi_size;
  PetscInt d_nz,o_nz;

  PetscInt lstate,sign;
  PetscScalar tmp_val;

  N = 1<<L;

  MPI_Comm_rank(PETSC_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&mpi_size);

  ierr = MatCreate(PETSC_COMM_WORLD,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);

  nrows = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD,&nrows,&N);

  /* find how many spins are local, for allocation purposes */
  local_bits = 0;
  while(nrows >>= 1) ++local_bits;

  /* now count how many terms are not on our processor */

  /* this creates something that looks like 111110000000 */
  nonlocal_mask = (-1) << local_bits;

  d_nz = o_nz = 0;
  for (i=0;i<nterms;++i) {
    /* only count each element once, even though
       there might be a few terms that contribute to it */
    if (i>0 && masks[i-1] == masks[i]) continue;

    if (masks[i] & nonlocal_mask) ++o_nz;
    else ++d_nz;
  }

  /* it seems from the documentation you can just call both of these and it will use the right one */
  ierr = MatSeqAIJSetPreallocation(*A,d_nz,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*A,d_nz,NULL,o_nz,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(*A,&Istart,&Iend);CHKERRQ(ierr);

  /* this is where the magic happens */
  for (state=Istart;state<Iend;++state) {
    for (i=0;i<nterms;) {
      lstate = state ^ masks[i];
      tmp_val = 0;
      /* sum all terms for this matrix element */
      do {
        /* this requires gcc builtins */
        sign = 1 - 2*(__builtin_popcount(state & signs[i]) % 2);
        tmp_val += sign * coeffs[i];
        ++i;
      } while (i<nterms && masks[i-1] == masks[i]);

      /* the elements must not be repeated or else INSERT_VALUES is wrong! */
      /* I could just use ADD_VALUES but if they are repeated there is a bug somewhere else */
      ierr = MatSetValues(*A,1,&lstate,1,&state,&tmp_val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return ierr;

}

#undef  __FUNCT__
#define __FUNCT__ "BuildMat_Shell"
PetscErrorCode BuildMat_Shell(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt N,n;
  shell_context *ctx;

  N = 1<<L;

  n = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD,&n,&N);

  ierr = BuildContext(L,nterms,masks,signs,coeffs,&ctx);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,n,n,N,N,ctx,A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))MatMult_Shell);
  ierr = MatShellSetOperation(*A,MATOP_NORM,(void(*)(void))MatNorm_Shell);

  return ierr;

}

static inline PetscInt map_back(PetscInt *choose_array,PetscInt s)
{
  /* TODO */
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "MatMult_Shell"
PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscInt lstate,lstate_idx,state,i,Istart,Iend,block_start,sign,m,s;
  PetscInt *sub_map;
  const PetscScalar *x_array,*sub_x;
  PetscScalar c;
  shell_context *ctx;

  /* cache */
  PetscInt *indices,cache_idx,cache_idx_max;
  PetscScalar *values;

  ierr = PetscMalloc(sizeof(PetscInt) * VECSET_CACHE_SIZE,&indices);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar) * VECSET_CACHE_SIZE,&values);CHKERRQ(ierr);

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  /* clear out the b vector */
  ierr = VecSet(b,0);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&x_array);CHKERRQ(ierr);

  for (i=0;i<ctx->nterms;++i) {

    m = ctx->masks[i];
    s = ctx->signs[i];
    c = ctx->coeffs[i];

    for (block_start=Istart;
         block_start<Iend;
         block_start+=VECSET_CACHE_SIZE) {

      cache_idx_max = PetscMin(Iend-block_start,VECSET_CACHE_SIZE);

      // so that we can index from 0
      sub_x = x_array + (block_start-Istart);
      if (ctx->state_map != NULL) {
        sub_map = ctx->state_map + block_start;
      }

      /*
       * the goal is that the following for loops will be vectorized by the
       * compiler. Everything should be pretty trivially
       * vectorizable, though __builtin_popcount may or may not be depending
       * on the compiler--maybe it will be worth it to write a vectorizable
       * popcount. (I guess I can check the assembly to see if it vectorizes)
       */
      if (ctx->state_map == NULL) {
        for (cache_idx=0;cache_idx<cache_idx_max;++cache_idx) {
          state = block_start + cache_idx;
          indices[cache_idx] = state ^ m;
          sign = 1 - 2*(__builtin_popcount(state & s)&1);
          values[cache_idx] = sub_x[cache_idx] * sign * c;
        }
      }
      else {
        /* this hopefully is vectorized */
        for (cache_idx=0;cache_idx<cache_idx_max;++cache_idx) {
          state = sub_map[cache_idx];
          sign = 1 - 2*(__builtin_popcount(state & s)&1);
          values[cache_idx] = sub_x[cache_idx] * sign * c;
        }
        /* it's not clear how to make this next part vectorizable.
         * I leave it for now, and perhaps figure out if I
         * can cleverly make it vectorizable in the future
         */
        for (cache_idx=0;cache_idx<cache_idx_max;++cache_idx) {
          lstate = sub_map[cache_idx] ^ m;
          lstate_idx = map_back(ctx->choose_array,lstate);
          indices[cache_idx] = lstate_idx;
        }
      }
      /*
       * here will need to add some functions for spin > 1/2 systems
       * they will involve more complicated logic for how to compute
       * what the matrix element is
       */
      ierr = VecSetValues(b,cache_idx_max,indices,values,ADD_VALUES);CHKERRQ(ierr);
    }
    /*
     * For some operators the stash of values waiting to be sent to other processes
     * may grow to a huge number if we wait too long. Maybe we should
     * call VecAssembly* functions here and clear out the stash, to save some memory
     */
  }
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);

  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "MatNorm_Shell"
PetscErrorCode MatNorm_Shell(Mat A,NormType type,PetscReal *nrm)
{
  PetscErrorCode ierr;
  PetscInt m,idx,state,N,i,sign,map;
  PetscScalar csum;
  PetscReal sum,max_sum;
  shell_context *ctx;

  if (type != NORM_INFINITY) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Only NORM_INFINITY is implemented for shell matrices.");
  }

  /* TODO: this whole computation can be done in parallel, with
     each process doing a chunk of the matrix. it wouldn't be a
     totally crazy speedup since we only do this once. but still
     probably would be helpful, especially for really big matrices */

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  /*
    keep the norm cached so we don't have to compute it all the time.
    if we already have it, just return it
  */
  if (ctx->nrm != -1) {
    (*nrm) = ctx->nrm;
    return ierr;
  }

  N = 1<<ctx->L;
  max_sum = 0;
  map = ctx->state_map != NULL;
  for (idx=0;idx<N;++idx) {
    sum = 0;
    for (i=0;i<ctx->nterms;) {
      csum = 0;
      m = ctx->masks[i];
      if (map) {
        state = ctx->state_map[idx];
      }
      else state = idx;
      /* sum all terms for this matrix element */
      do {
        /* this requires gcc builtins */
        sign = 1 - 2*(__builtin_popcount(state & ctx->signs[i]) % 2);
        csum += sign * ctx->coeffs[i];
        ++i;
      } while (i<ctx->nterms && m == ctx->masks[i]);

      sum += PetscAbsComplex(csum);
    }
    if (sum > max_sum) {
      max_sum = sum;
    }
  }

  ctx->nrm = (*nrm) = max_sum;

  return ierr;

}

#undef  __FUNCT__
#define __FUNCT__ "BuildContext"
PetscErrorCode BuildContext(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,shell_context **ctx_p)
{
  PetscErrorCode ierr;
  shell_context *ctx;
  PetscInt i;

  ierr = PetscMalloc(sizeof(shell_context),ctx_p);CHKERRQ(ierr);
  ctx = (*ctx_p);

  ctx->L = L;
  ctx->nterms = nterms;
  ctx->nrm = -1;
  ctx->gpu = PETSC_FALSE;

  /* TODO: implement this logic */
  ctx->state_map = NULL;

  /* we need to keep track of this stuff on our own. the numpy array might get garbage collected */
  ierr = PetscMalloc(sizeof(PetscInt)*nterms,&(ctx->masks));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nterms,&(ctx->signs));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*nterms,&(ctx->coeffs));CHKERRQ(ierr);

  for (i=0;i<nterms;++i) {
    ctx->masks[i] = masks[i];
    ctx->signs[i] = signs[i];
    ctx->coeffs[i] = coeffs[i];
  }

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "DestroyContext"
PetscErrorCode DestroyContext(Mat A)
{
  PetscErrorCode ierr;
  shell_context *ctx;

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  ierr = PetscFree(ctx->masks);CHKERRQ(ierr);
  ierr = PetscFree(ctx->signs);CHKERRQ(ierr);
  ierr = PetscFree(ctx->coeffs);CHKERRQ(ierr);

  ierr = PetscFree(ctx);CHKERRQ(ierr);

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "ReducedDensityMatrix"
PetscErrorCode ReducedDensityMatrix(PetscInt L,
                                    Vec x,
                                    PetscInt cut_size,
                                    PetscBool fillall,
                                    PetscScalar* m)
{
  const PetscScalar *x_array;
  PetscInt i,j,k,jmax;
  PetscInt cut_N,tr_N,tr_size;
  PetscScalar a,b;
  PetscErrorCode ierr;

  /* compute sizes of things */
  tr_size = L - cut_size; /* in case L is odd */
  cut_N = 1 << cut_size;
  tr_N = 1 << tr_size;

  /*
    Fill the reduced density matrix!

    We only need to fill the lower triangle for the
    purposes of eigensolving for entanglement entropy.
    This behavior is controlled by "fillall".
  */
  ierr = VecGetArrayRead(x,&x_array);CHKERRQ(ierr);

  for (i=0;i<cut_N;++i) {

    if (fillall) jmax = cut_N;
    else jmax = i+1;

    for (j=0;j<jmax;++j) {

      for (k=0;k<tr_N;++k) {
        a = x_array[(i<<tr_size) + k];
        b = x_array[(j<<tr_size) + k];

        m[i*cut_N + j] += a*PetscConj(b);
      }
    }
  }

  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);

  return ierr;

}
