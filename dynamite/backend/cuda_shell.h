
#include <petscmat.h>
#include <petsccuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include "shellcontext.h"

#define GPU_BLOCK_SIZE 128
#define GPU_BLOCK_NUM 128

PetscErrorCode BuildContext_CUDA(PetscInt L,
                                 PetscInt nterms,
                                 PetscInt* masks,
                                 PetscInt* signs,
                                 PetscScalar* coeffs,
                                 shell_context **ctx_p);

PetscErrorCode DestroyContext_CUDA(Mat A);

PetscErrorCode BuildMat_CUDAShell(PetscInt L,
				  PetscInt nterms,
				  PetscInt* masks,
				  PetscInt* signs,
				  PetscScalar* coeffs,
				  Mat *A);

PetscErrorCode MatNorm_CUDAShell(Mat A,NormType type,PetscReal *nrm);
PetscErrorCode MatMult_CUDAShell(Mat M,Vec x,Vec b);

__global__ void device_MatMult_Shell(PetscInt size,
                                     PetscInt* masks,
                                     PetscInt* signs,
                                     PetscScalar* coeffs,
                                     PetscInt nterms,
                                     const PetscScalar* xarray,
                                     PetscScalar* barray);

__global__ void device_MatNorm_Shell(PetscInt size,
                                     PetscInt* masks,
                                     PetscInt* signs,
                                     PetscScalar* coeffs,
                                     PetscInt nterms,
                                     PetscReal* d_maxs);

