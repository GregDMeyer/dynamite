#pragma once

#include <slepcmfn.h>
#include "bsubspace_impl.h"

/* allow us to set many values at once */
#define VECSET_CACHE_SIZE 2048
#define ITER_CUTOFF 8
#define LKP_SIZE (1<<6)
#define LKP_MASK (LKP_SIZE-1)
#define intmin(a,b) ((a)^(((a)^(b))&(((a)<(b))-1)))

#define TERM_REAL(mask, sign) (!(__builtin_parity((mask) & (sign))))

typedef enum _shell_impl {
  NO_SHELL,
  CPU_SHELL,
  GPU_SHELL
} shell_impl;

typedef struct _msc_t {
  PetscInt nmasks;
  PetscInt* masks;
  PetscInt* mask_offsets;
  PetscInt* signs;
  PetscScalar* coeffs;
} msc_t;

typedef struct _shell_context {
  PetscInt nmasks;
  PetscInt* masks;
  PetscInt* mask_offsets;
  PetscInt* signs;
  PetscReal* real_coeffs;     // we store only the real or complex part -- whichever is nonzero
  void *left_subspace_data;
  void *right_subspace_data;
  PetscReal nrm;
} shell_context;

PetscErrorCode BuildMat(const msc_t *msc, subspaces_t *subspaces, shell_impl shell, Mat *A);

/* define a type for context destroying functions, and we keep that in the context */
// TODO

PetscErrorCode BuildContext(const msc_t *msc,
                            const void* left_subspace_data,
                            const void* right_subspace_data,
                            shell_context **ctx_p);
PetscErrorCode DestroyContext(Mat A);

PetscErrorCode ReducedDensityMatrix(PetscInt L,Vec x,PetscInt cut_size,PetscBool fillall,PetscScalar* m);
