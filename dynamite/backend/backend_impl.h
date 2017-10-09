#ifndef BACKEND_H
#define BACKEND_H

#include <slepcmfn.h>
#include "shellcontext.h"

/* allow us to set many values at once */
#define VECSET_CACHE_SIZE 2048

PetscErrorCode BuildMat_Full(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,Mat *A);
PetscErrorCode BuildMat_Shell(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,Mat *A);

PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec b);

PetscErrorCode MatNorm_Shell(Mat A,NormType type,PetscReal *nrm);

PetscErrorCode BuildContext(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,shell_context **ctx_p);
PetscErrorCode DestroyContext(Mat A);

PetscErrorCode ReducedDensityMatrix(PetscInt L,Vec x,PetscInt cut_size,PetscBool fillall,PetscScalar* m);

#endif /* !BACKEND_H */
