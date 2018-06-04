
#include "cuda_shell_private.h"

extern "C" {
#include "cuda_shell.h"
}

PetscErrorCode BuildMat_CUDAShell(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt N,n;
  shell_context *ctx;

  N = 1<<L;

  n = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD,&n,&N);

  ierr = BuildContext_CUDA(L,nterms,masks,signs,coeffs,&ctx);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,n,n,N,N,ctx,A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))MatMult_CUDAShell);
  ierr = MatShellSetOperation(*A,MATOP_NORM,(void(*)(void))MatNorm_CUDAShell);
  ierr = MatShellSetOperation(*A,MATOP_CREATE_VECS,(void(*)(void))MatCreateVecs_CUDAShell);

  return ierr;
}

PetscErrorCode BuildContext_CUDA(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,shell_context **ctx_p)
{
  PetscErrorCode ierr;
  cudaError_t err;
  shell_context *ctx;

  ierr = PetscMalloc(sizeof(shell_context),ctx_p);CHKERRQ(ierr);
  ctx = (*ctx_p);

  ctx->L = L;
  ctx->nterms = nterms;
  ctx->nrm = -1;
  ctx->gpu = PETSC_TRUE;

  err = cudaMalloc((void **) &(ctx->masks),  sizeof(PetscInt)*nterms);CHKERRCUDA(err);
  err = cudaMalloc((void **) &(ctx->signs),  sizeof(PetscInt)*nterms);CHKERRCUDA(err);
  err = cudaMalloc((void **) &(ctx->coeffs), sizeof(PetscScalar)*nterms);CHKERRCUDA(err);

  err = cudaMemcpy(ctx->masks,masks,sizeof(PetscInt)*nterms,cudaMemcpyHostToDevice);CHKERRCUDA(err);
  err = cudaMemcpy(ctx->signs,signs,sizeof(PetscInt)*nterms,cudaMemcpyHostToDevice);CHKERRCUDA(err);
  err = cudaMemcpy(ctx->coeffs,coeffs,sizeof(PetscScalar)*nterms,cudaMemcpyHostToDevice);CHKERRCUDA(err);

  return ierr;
}

PetscErrorCode DestroyContext_CUDA(Mat A)
{
  PetscErrorCode ierr;
  cudaError_t err;
  shell_context *ctx;

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  err = cudaFree(ctx->masks);CHKERRCUDA(err);
  err = cudaFree(ctx->signs);CHKERRCUDA(err);
  err = cudaFree(ctx->coeffs);CHKERRCUDA(err);

  ierr = PetscFree(ctx);CHKERRQ(ierr);

  return ierr;
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

  ierr = MatShellGetContext(M,&ctx);CHKERRQ(ierr);

  ierr = VecCUDAGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(b,&barray);CHKERRQ(ierr);

  size = 1 << ctx->L;

  err = cudaThreadSynchronize();CHKERRCUDA(err);

  device_MatMult_Shell<<<GPU_BLOCK_NUM,GPU_BLOCK_SIZE>>>(size,
                                                         ctx->masks,
                                                         ctx->signs,
                                                         ctx->coeffs,
                                                         ctx->nterms,
                                                         xarray,
                                                         barray);

  err = cudaThreadSynchronize();CHKERRCUDA(err);

  ierr = VecCUDARestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(b,&barray);CHKERRQ(ierr);

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

  PetscScalar tmp,val;
  PetscInt state,ket,mask,next_mask,this_start,i;

  this_start = vec_start_index + threadIdx.x;

  /* only access mask from global memory once */

  /* on the gpu, unlike on parallel CPUs, we have access
   * to the whole vector from any processor. That's awesome
   * because it means that we can accumulate results by row
   * instead of by column, and only do a single memory write
   * per entry in the output vector. Then we don't have to worry
   * about atomic operations either!
   */

  for (ket=this_start; ket<vec_stop_index; ket += blockDim.x) {
    val = 0;
    mask = masks[0];
    for (i=0;i<nterms;) {
      tmp = 0;
      state = ket ^ mask;
      /* sum all terms for this matrix element */
      do {
#if defined(PETSC_USE_64BIT_INDICES)
        tmp += __popcll(state & signs[i])%2 ? -coeffs[i] : coeffs[i];
#else
        tmp += __popc(state & signs[i])%2 ? -coeffs[i] : coeffs[i];
#endif
        ++i;
        if (i == nterms) break;
        next_mask = masks[i];
      } while (mask == next_mask);
      /* this can be optimized by keeping track of # of terms per matrix element.
         I think that should actually make it a lot faster because it gets rid of
         a significant chunk of the memory reads */
      val += tmp * xarray[state];
      mask = next_mask;
    }
    barray[ket] = val;
  }
}

PetscErrorCode MatNorm_CUDAShell(Mat A,NormType type,PetscReal *nrm)
{
  PetscErrorCode ierr;
  cudaError_t err;
  shell_context *ctx;

  PetscReal *d_maxs,*h_maxs;
  PetscInt i,N;

  if (type != NORM_INFINITY) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Only NORM_INFINITY is implemented for shell matrices.");
  }

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  /*
    keep the norm cached so we don't have to compute it all the time.
    if we already have it, just return it
  */
  if (ctx->nrm != -1) {
    (*nrm) = ctx->nrm;
    return ierr;
  }

  err = cudaMalloc((void **) &d_maxs,sizeof(PetscReal)*GPU_BLOCK_NUM);CHKERRCUDA(err);
  ierr = PetscMalloc(sizeof(PetscReal)*GPU_BLOCK_NUM,&h_maxs);CHKERRQ(ierr);

  N = 1<<ctx->L;

  device_MatNorm_Shell<<<GPU_BLOCK_NUM,GPU_BLOCK_SIZE,sizeof(PetscReal)*GPU_BLOCK_SIZE>>>(N,ctx->masks,ctx->signs,ctx->coeffs,ctx->nterms,d_maxs);

  err = cudaThreadSynchronize();CHKERRCUDA(err);

  err = cudaMemcpy(h_maxs,d_maxs,sizeof(PetscReal)*GPU_BLOCK_NUM,cudaMemcpyDeviceToHost);CHKERRCUDA(err);

  /* now do max of h_maxs */
  (*nrm) = 0;
  for (i=0;i<GPU_BLOCK_NUM;++i) {
    if (h_maxs[i] > (*nrm)) (*nrm) = h_maxs[i];
  }

  ctx->nrm = (*nrm);

  err = cudaFree(d_maxs);CHKERRCUDA(err);
  ierr = PetscFree(h_maxs);CHKERRQ(ierr);

  return ierr;
}

__global__ void device_MatNorm_Shell(PetscInt size,
                                     PetscInt* masks,
                                     PetscInt* signs,
                                     PetscScalar* coeffs,
                                     PetscInt nterms,
                                     PetscReal *d_maxs)
{
  extern __shared__ PetscReal threadmax[];

  /* the following four lines come from the PETSc cuda source */
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscReal sum,v1,v2;
  PetscScalar csum;
  PetscInt state, i, mask, next_mask;

  /* first find this thread's max and put it in threadmax */

  threadmax[threadIdx.x] = 0;
  for (state=vec_start_index+threadIdx.x;state<vec_stop_index;state += blockDim.x) {
    sum = 0;
    for (i=0;i<nterms;) {
      csum = 0;
      mask = masks[i];
      /* sum all terms for this matrix element */
      do {
#if defined(PETSC_USE_64BIT_INDICES)
        csum += __popcll(state & signs[i])%2 ? -coeffs[i] : coeffs[i];
#else
        csum += __popc(state & signs[i])%2 ? -coeffs[i] : coeffs[i];
#endif
        ++i;
        if (i >= nterms) break;
        next_mask = masks[i];
      } while (mask == next_mask);

      sum += abs(csum);
    }
    if (sum > threadmax[threadIdx.x]) {
      threadmax[threadIdx.x] = sum;
    }
  }
  __syncthreads();

  /* now do the coolest reduce ever on the shared memory and hand it off to CPU */

  for (i=1; i<blockDim.x; i*=2) {
    if (threadIdx.x % (2*i) == 0) {
      v1 = threadmax[threadIdx.x];
      v2 = threadmax[threadIdx.x + i];
      threadmax[threadIdx.x] = v1>v2 ? v1 : v2;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) d_maxs[blockIdx.x] = threadmax[0];
}

PetscErrorCode MatCreateVecs_CUDAShell(Mat mat, Vec *right, Vec *left)
{
  PetscErrorCode ierr;
  PetscInt N;

  ierr = MatGetSize(mat,&N,NULL);CHKERRQ(ierr);

  if (right) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,PETSC_DECIDE,N);CHKERRQ(ierr);
    ierr = VecSetFromOptions(*right);
  }
  if (left) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,PETSC_DECIDE,N);CHKERRQ(ierr);
    ierr = VecSetFromOptions(*left);
  }

  return 0;
}
