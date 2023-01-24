#pragma once

#include <slepcmfn.h>
#include "bsubspace_impl.h"
#include "shell_context.h"

/* allow us to set many values at once */
#define BLOCK_SIZE 2048

#define intmin(a,b) ((a)^(((a)^(b))&(((a)<(b))-1)))

#ifdef PETSC_USE_64BIT_INDICES
  #define builtin_ctz __builtin_ctzl
  #define builtin_parity __builtin_parityl
#else
  #define builtin_ctz __builtin_ctz
  #define builtin_parity __builtin_parity
#endif

// the following two are to pass to the python side

#ifdef PETSC_USE_COMPLEX
  #define DNM_PETSC_COMPLEX 1
#else
  #define DNM_PETSC_COMPLEX 0
#endif

#ifdef PETSC_HAVE_CUDA
  #define DNM_PETSC_CUDA 1
#else
  #define DNM_PETSC_CUDA 0
#endif

#define TERM_REAL(mask, sign) (!(builtin_parity((mask) & (sign))))

typedef enum _shell_impl {
  NO_SHELL,
  CPU_SHELL,
  GPU_SHELL
} shell_impl;

PetscErrorCode BuildMat(const msc_t *msc, subspaces_t *subspaces, shell_impl shell, int xparity, Mat *A);

PetscErrorCode CheckConserves(const msc_t *msc, subspaces_t *subspaces, int xparity, PetscInt *result);

PetscErrorCode BuildContext(const msc_t *msc,
                            const void* left_subspace_data,
                            const void* right_subspace_data,
                            shell_context **ctx_p);
PetscErrorCode DestroyContext(Mat A);

PetscErrorCode ReducedDensityMatrix(
  Vec vec,
  PetscInt sub_type,
  void* sub_data_p,
  PetscInt keep_size,
  PetscInt* keep,
  PetscBool triang,
  PetscInt rtn_dim,
  PetscScalar* rtn
);
