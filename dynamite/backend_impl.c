
#include "backend_impl.h"

/* allow us to set many values at once */
#define VECSET_CACHE_SIZE 1000

#undef  __FUNCT__
#define __FUNCT__ "BuildMat_Full"
PetscErrorCode BuildMat_Full(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt N,i,state,Istart,Iend,mpi_rank,mpi_size,nrows,local_bits,nonlocal_mask;
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
  for (int i=0;i<nterms;++i) {
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
      ierr = MatSetValues(*A,1,&state,1,&lstate,&tmp_val,INSERT_VALUES);CHKERRQ(ierr);
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

  nrows = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD,&nrows,&N);

  ierr = BuildContext(L,nterms,masks,signs,coeffs,&ctx);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,m,n,N,N,ctx,&A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H,MATOP_MULT,(void(*)(void))MatMult_Shell);
  ierr = MatShellSetOperation(H,MATOP_NORM,(void(*)(void))MatNorm_Shell);

  return ierr;

}

#undef  __FUNCT__
#define __FUNCT__ "MatMult_Shell"
PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscInt state,lstate,i,sign;
  PetscScalar tmp_val,*x_array;
  shell_context *ctx;

  /* cache */
  PetscInt *indices,cache_index;
  PetscScalar *values;

  /* create a cache to keep our values in */
  ierr = PetscMalloc(sizeof(PetscInt) * VECSET_CACHE_SIZE,&indices);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar) * VECSET_CACHE_SIZE,&values);CHKERRQ(ierr);

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  /* clear out the b vector */
  ierr = VecSet(b,0);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&x_array);CHKERRQ(ierr);

  /* TODO: get b array and use that directly to set local values */
  cache_index = 0;
  for (state=Istart;state<Iend;++state) {
    for (i=0;i<ctx->nterms;) {
      indices[cache_index] = state ^ ctx->masks[i];
      tmp_val = 0;
      /* sum all terms for this matrix element */
      do {
        /* this requires gcc builtins */
        sign = 1 - 2*(__builtin_popcount(state & ctx->signs[i]) % 2);
        tmp_val += sign * ctx->coeffs[i];
        ++i;
      } while (i<ctx->nterms && ctx->masks[i-1] == ctx->masks[i]);

      values[cache_index] = tmp_val * x_array[state-Istart];

      ++cache_index;

      if (cache_index >= VECSET_CACHE_SIZE) {
        ierr = VecSetValues(b,cache_index,indices,values,ADD_VALUES);CHKERRQ(ierr);
        cache_index = 0;
      }
    }
  }

  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);

  return ierr;

}

#undef  __FUNCT__
#define __FUNCT__ "MatNorm_Shell"
PetscErrorCode MatNorm_Shell(Mat A,NormType type,PetscReal *nrm)
{
  PetscErrorCode ierr;
  PetscInt state,N;
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
  for (state=0;state<;++state) {
    sum = 0
    for (i=0;i<ctx->nterms;) {
      csum = 0;
      /* sum all terms for this matrix element */
      do {
        /* this requires gcc builtins */
        sign = 1 - 2*(__builtin_popcount(state & ctx->signs[i]) % 2);
        csum += sign * ctx->coeffs[i];
        ++i;
      } while (i<ctx->nterms && ctx->masks[i-1] == ctx->masks[i]);

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
  ctx = (*ctx_p)

  ctx->L = L;
  ctx->nterms = nterms;
  ctx->nrm = -1;

  /* we need to keep track of this stuff on our own. the numpy array might get garbage collected */
  ierr = PetscMalloc(sizeof(PetscInt)*nterms,ctx->masks);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*nterms,ctx->signs);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*nterms,ctx->coeffs);CHKERRQ(ierr);

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
}
