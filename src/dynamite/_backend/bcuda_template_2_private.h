
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <petscmat.h>
#include "bcuda_template_2.h"

PetscErrorCode C(BuildContext_CUDA,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const C(data,LEFT_SUBSPACE)* left_subspace_data,
  const C(data,RIGHT_SUBSPACE)* right_subspace_data,
  shell_context **ctx_p);

PetscErrorCode C(MatDestroyCtx_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A);

PetscErrorCode C(MatMult_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b);

__global__ void C(device_MatMult,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  PetscInt row_start,
  PetscInt row_end,
  PetscInt* masks,
  PetscInt* mask_offsets,
  PetscInt* signs,
  PetscReal* real_coeffs,
  PetscInt nmasks,
  C(data,LEFT_SUBSPACE) *left_subspace_data,
  C(data,RIGHT_SUBSPACE) *right_subspace_data,
  PetscReal* diag,
  const PetscScalar* xarray,
  PetscScalar* barray);

PetscErrorCode C(MatNorm_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, NormType type, PetscReal *nrm);

__global__ void C(device_MatNorm,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  PetscInt size,
  PetscInt* masks,
  PetscInt* mask_offsets,
  PetscInt* signs,
  PetscReal* real_coeffs,
  PetscInt nmasks,
  C(data,LEFT_SUBSPACE) *left_subspace_data,
  C(data,RIGHT_SUBSPACE) *right_subspace_data,
  PetscReal *d_maxs);
