
#include "bcuda_template_1_private.h"

PetscErrorCode C(PrecomputeDiagonal_GPU,SUBSPACE)(Mat A)
{
  PetscInt size;
  shell_context *ctx;
  PetscCall(MatShellGetContext(A, &ctx));

  PetscCall(MatGetSize(A, &size, PETSC_NULLPTR));

  PetscCallCUDA(cudaMalloc((void **) &(ctx->diag), sizeof(PetscReal)*size));

  PetscCallCUDA(cudaDeviceSynchronize());

  C(device_PrecomputeDiagonal,SUBSPACE)<<<GPU_BLOCK_NUM,GPU_BLOCK_SIZE>>>(
    size,
    ctx->mask_offsets,
    ctx->signs,
    ctx->real_coeffs,
    (C(data,SUBSPACE)*) ctx->right_subspace_data,
    ctx->diag);

  PetscCallCUDA(cudaDeviceSynchronize());

  return 0;
}

__global__ void C(device_PrecomputeDiagonal,SUBSPACE)(
  PetscInt size,
  PetscInt* mask_offsets,
  PetscInt* signs,
  PetscReal* real_coeffs,
  C(data,SUBSPACE) *subspace_data,
  PetscReal* diag)
{

  /* the following four lines come from the PETSc cuda source */
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscReal val;
  PetscReal sign;
  PetscInt state, row_idx, term_idx, this_start;

  this_start = vec_start_index + threadIdx.x;

  for (row_idx=this_start; row_idx<vec_stop_index; row_idx+=blockDim.x) {
    state = C(I2S_CUDA,SUBSPACE)(row_idx, subspace_data);
    val = 0;

    /* sum all terms for this matrix element */
    for (term_idx=0; term_idx<mask_offsets[1]; ++term_idx) {
#if defined(PETSC_USE_64BIT_INDICES)
      sign = __popcll(state & signs[term_idx])&1;
#else
      sign = __popc(state & signs[term_idx])&1;
#endif
      sign = 1 - 2*sign;
      val += sign * real_coeffs[term_idx];
    }
    diag[row_idx] = val;
  }
}
