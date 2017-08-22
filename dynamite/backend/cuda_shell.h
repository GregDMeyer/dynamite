
#include <PetscMat.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

#define GPU_BLOCK_SIZE 128
#define GPU_BLOCK_NUM 128

typedef struct _shell_context_CUDA {
    PetscInt L;
    PetscInt nterms;
    PetscInt *masks;
    PetscInt *signs;
    PetscScalar *coeffs;
    PetscReal nrm;
} shell_context_CUDA;

PetscErrorCode BuildContext_CUDA(PetscInt L,
                                 PetscInt nterms,
                                 PetscInt* masks,
                                 PetscInt* signs,
                                 PetscScalar* coeffs,
                                 shell_context **ctx_p);

PetscErrorCode DestroyContext_CUDA(Mat A);

PetscErrorCode MatMult_CUDAShell(Mat M,Vec x,Vec b);