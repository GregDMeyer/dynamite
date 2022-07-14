
#include <petsc.h>
#include <bpetsc_impl.h>
#include <assert.h>

#define LEFT_SUBSPACE Full
  #define RIGHT_SUBSPACE Full
    #include <buildmat_template.h>
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE

int main(int argc, char const *argv[]) {

  PetscErrorCode ierr;
  PetscScalar* x_array;
  PetscScalar* b_array;
  shell_context ctx;
  data_Full subspace_data;

  /* test identity */
  ctx.nmasks = 1;
  ierr = PetscMalloc1(1, &(ctx.masks));CHKERRQ(ierr);
  ierr = PetscMalloc1(2, &(ctx.mask_offsets));CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &(ctx.signs));CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &(ctx.real_coeffs));CHKERRQ(ierr);

  ctx.masks[0] = 0;
  ctx.mask_offsets[0] = 0;
  ctx.mask_offsets[1] = 1;
  ctx.signs[0] = 0;
  ctx.real_coeffs[0] = 1.;

  subspace_data.L = -1;
  ctx.left_subspace_data = &subspace_data;
  ctx.right_subspace_data = &subspace_data;

  #define DIM 100

  ierr = PetscMalloc1(DIM, &x_array);CHKERRQ(ierr);
  ierr = PetscMalloc1(DIM, &b_array);CHKERRQ(ierr);

  for (PetscInt i = 0; i < DIM; ++i) {
    x_array[i] = i;
    b_array[i] = 0;
  }

  MatMult_CPU_kernel_Full_Full(x_array, b_array, &ctx, 0, DIM, 0, DIM);

  for (PetscInt i = 0; i < DIM; ++i) {
    assert(b_array[i] == i);
  }

  ierr = PetscFree(x_array);CHKERRQ(ierr);
  ierr = PetscFree(b_array);CHKERRQ(ierr);

  #undef DIM

  ierr = PetscFree(ctx.masks);CHKERRQ(ierr);
  ierr = PetscFree(ctx.mask_offsets);CHKERRQ(ierr);
  ierr = PetscFree(ctx.signs);CHKERRQ(ierr);
  ierr = PetscFree(ctx.real_coeffs);CHKERRQ(ierr);

  return 0;
}
