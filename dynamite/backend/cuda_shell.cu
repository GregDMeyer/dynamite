
#include "cuda_shell.h"

PetscErrorCode BuildContext_CUDA(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,shell_context **ctx_p)
{
  PetscErrorCode ierr;
  cudaError_t err;
  shell_context *ctx;
  PetscInt i;

  ierr = PetscMalloc(sizeof(shell_context),ctx_p);CHKERRQ(ierr);
  ctx = (*ctx_p);

  ctx->L = L;
  ctx->nterms = nterms;
  ctx->nrm = -1;

  err = cudaMalloc((void **) &(ctx->masks),  sizeof(PetscInt)*nterms);CHKERRCUDA(err);
  err = cudaMalloc((void **) &(ctx->signs),  sizeof(PetscInt)*nterms);CHKERRCUDA(err);
  err = cudaMalloc((void **) &(ctx->coeffs), sizeof(PetscScalar)*nterms);CHKERRCUDA(err);

  err = cudaMemcpy(masks,ctx->masks,cudaMemcpyHostToDevice);CHKERRCUDA(err);
  err = cudaMemcpy(signs,ctx->signs,cudaMemcpyHostToDevice);CHKERRCUDA(err);
  err = cudaMemcpy(coeffs,ctx->coeffs,cudaMemcpyHostToDevice);CHKERRCUDA(err);

  return ierr;
}

PetscErrorCode DestroyContext(Mat A)
{
  PetscErrorCode ierr;
  cudaError_t err;
  shell_context *ctx;

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  err = cudaFree(ctx->masks);CHKERRCUDA(err);
  err = cudaFree(ctx->signs);CHKERRCUDA(err);
  err = cudaFree(ctx->coeffs);CHKERRCUDA(err);

  ierr = PetscFree(ctx);CHKERRQ(ierr);

  return ierr
}

PetscErrorCode MatMult_CUDAShell(Mat M,Vec x,Vec b)
{
  PetscErrorCode ierr;
  cudaError_t err;
  shell_context *ctx;

  const PetscScalar* xarray;
  PetscScalar* barray;
  PetscInt size;

  ierr = VecSet(b,0);CHKERRQ(ierr);

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  ierr = VecCUDAGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(b,&barray);CHKERRQ(ierr);

  size = 1 << ctx->L;

  device_MatMult_Shell<<<GPU_BLOCK_NUM,GPU_BLOCK_SIZE>>>(size,
                                                         ctx->masks,
                                                         ctx->signs,
                                                         ctx->coeffs,
                                                         ctx->nterms,
                                                         xarray,
                                                         barray);

  ierr = VecRestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArrayReadWrite(b,&barray);CHKERRQ(ierr);

  return ierr;
}

__global__ void device_MatMult_Shell(PetscInt size,
                                     PetscInt* masks,
                                     PetscInt* signs,
                                     PetscScalar* coeffs,
                                     PetscInt nterms,
                                     const PetscScalar* xarray,
                                     PetscScalar* barray)
{

  /* the following four lines come from the PETSc cuda source */
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar tmp;
  PetscInt ket,mask,next_mask,this_start;

  this_start = vec_start_index + threadIdx.x;

  /* only access mask from global memory once */
  mask = ctx->masks[this_start];
  for (state=this_start; state<vec_stop_index; state += blockIdx.x) {
    for (i=0;i<ctx->nterms;) {
      ket = state ^ mask;
      tmp = 0;
      /* sum all terms for this matrix element */
      do {
        /* if __builtin_popcount doesn't exist in nvcc, need to do it manually */
        sign = 1 - 2*(__builtin_popcount(state & ctx->signs[i]) % 2);
        tmp += sign * ctx->coeffs[i];
        ++i;
        if (i == ctx->nterms) break;
        next_mask = ctx->masks[i];
      } while (mask == next_mask);
      /* this can be optimized by keeping track of # of terms per matrix element.
         I think that should actually make it a lot faster because it gets rid of
         a significant chunk of the global memory reads */

      barray[ket] = tmp_val * xarray[state];
      mask = next_mask;
    }
  }
}