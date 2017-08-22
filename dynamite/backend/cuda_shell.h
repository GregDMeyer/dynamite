
#pragma once
#include <petscmat.h>
#include <petsccuda.h>
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
