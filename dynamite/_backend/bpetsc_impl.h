#pragma once

#include <slepcmfn.h>
#include "shellcontext.h"
#include "bsubspace_impl.h"

/* allow us to set many values at once */
#define VECSET_CACHE_SIZE 2048
#define ITER_CUTOFF 8
#define LKP_SIZE (1<<6)
#define LKP_MASK (LKP_SIZE-1)
#define intmin(a,b) ((a)^(((a)^(b))&(((a)<(b))-1)))

typedef struct _msc_t {
  PetscInt nmasks;
  PetscInt* masks;
  PetscInt* mask_offsets;
  PetscInt* signs;
  PetscScalar* coeffs;
} msc_t;

PetscErrorCode BuildMat(const msc_t *msc, subspaces_t *subspaces, Mat *A);

PetscErrorCode BuildMat_Shell(PetscInt L, const msc_t *msc,
                              const void *left_subspace_data,
                              const void *right_subspace_data,
                              Mat *A);

PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec b);

PetscErrorCode MatNorm_Shell(Mat A,NormType type,PetscReal *nrm);

PetscErrorCode BuildContext(PetscInt L,PetscInt nterms,
                            const PetscInt* masks,
                            const PetscInt* signs,
                            const PetscScalar* coeffs,
                            subspaces_t s,
                            shell_context **ctx_p);
PetscErrorCode DestroyContext(Mat A);

PetscErrorCode ReducedDensityMatrix(PetscInt L,Vec x,PetscInt cut_size,PetscBool fillall,PetscScalar* m);
