
#include "bcuda_template_1.h"

__global__ void C(device_PrecomputeDiagonal,SUBSPACE)(
  PetscInt size,
  PetscInt* mask_offsets,
  PetscInt* signs,
  PetscReal* real_coeffs,
  C(data,SUBSPACE) *right_subspace_data,
  PetscReal* diag);
