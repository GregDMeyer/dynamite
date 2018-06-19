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

PetscErrorCode BuildMat_Full(PetscInt L,PetscInt nterms,
                             const PetscInt* masks,
                             const PetscInt* signs,
                             const PetscScalar* coeffs,
                             Subspaces s,
                             Mat *A);
PetscErrorCode BuildMat_Shell(PetscInt L,PetscInt nterms,
                              const PetscInt* masks,
                              const PetscInt* signs,
                              const PetscScalar* coeffs,
                              Subspaces s,
                              Mat *A);

PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec b);

PetscErrorCode MatNorm_Shell(Mat A,NormType type,PetscReal *nrm);

PetscErrorCode BuildContext(PetscInt L,PetscInt nterms,
                            const PetscInt* masks,
                            const PetscInt* signs,
                            const PetscScalar* coeffs,
                            Subspaces s,
                            shell_context **ctx_p);
PetscErrorCode DestroyContext(Mat A);

PetscErrorCode ReducedDensityMatrix(PetscInt L,Vec x,PetscInt cut_size,PetscBool fillall,PetscScalar* m);
