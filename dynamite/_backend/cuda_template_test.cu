
#include "bcuda_impl.h"

#ifdef PETSC_USE_64BIT_INDICES
  #define TERM_REAL(mask, sign) (!(__popcll((mask) & (sign))))
#else
  #define TERM_REAL(mask, sign) (!(__popc((mask) & (sign))))
#endif

/* subspace functions for the GPU */

// TODO: use constant memory for the subspace data
// TODO: should pass each subspace data member individually to kernel?

PetscErrorCode CopySubspaceData_CUDA_Full(data_Full** out_p, const data_Full* in) {
  cudaError_t err;
  err = cudaMalloc((void **) out_p, sizeof(data_Full));CHKERRCUDA(err);
  err = cudaMemcpy(*out_p, in, sizeof(data_Full), cudaMemcpyHostToDevice);CHKERRCUDA(err);
  return 0;
}

PetscErrorCode DestroySubspaceData_CUDA_Full(data_Full* data) {
  cudaError_t err;
  err = cudaFree(data);CHKERRCUDA(err);
  return 0;
}

__device__ PetscInt S2I_CUDA_Full(PetscInt state, const data_Full* data) {
  return state;
}

__device__ PetscInt I2S_CUDA_Full(PetscInt idx, const data_Full* data) {
  return idx;
}

PetscErrorCode MatCreateVecs_GPU(Mat mat, Vec *right, Vec *left)
{
  PetscErrorCode ierr;
  PetscInt M, N;

  ierr = MatGetSize(mat, &M, &N);CHKERRQ(ierr);

  if (right) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right, PETSC_DECIDE, N);CHKERRQ(ierr);
    ierr = VecSetFromOptions(*right);
  }
  if (left) {
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left, PETSC_DECIDE, M);CHKERRQ(ierr);
    ierr = VecSetFromOptions(*left);
  }

  return 0;
}

#define LEFT_SUBSPACE Full
  #define RIGHT_SUBSPACE Full
    #include "bcuda_template.cu"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE
