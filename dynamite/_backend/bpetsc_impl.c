
#include "bpetsc_impl.h"

/*
 * For efficiency, we avoid going through cases of each subspace in the functions
 * defined in builmat_template.c. Instead, we just include the template multiple
 * times, using macros to define different functionality.
 */

#define SUBSPACE Full
  #include "bpetsc_template_1.c"
#undef SUBSPACE

#define SUBSPACE Parity
  #include "bpetsc_template_1.c"
#undef SUBSPACE

#define SUBSPACE Auto
  #include "bpetsc_template_1.c"
#undef SUBSPACE

#undef  __FUNCT__
#define __FUNCT__ "ReducedDensityMatrix"
PetscErrorCode ReducedDensityMatrix(
  Vec vec,
  PetscInt sub_type,
  void* sub_data_p,
  PetscInt keep_size,
  PetscInt* keep,
  PetscBool triang,
  PetscInt rtn_dim,
  PetscScalar* rtn
){
  PetscErrorCode ierr;
  switch (sub_type) {
    case FULL:
      ierr = rdm_Full(vec, sub_data_p, keep_size, keep, triang, rtn_dim, rtn);CHKERRQ(ierr);
      break;
    case PARITY:
      ierr = rdm_Parity(vec, sub_data_p, keep_size, keep, triang, rtn_dim, rtn);CHKERRQ(ierr);
      break;
    case AUTO:
      ierr = rdm_Auto(vec, sub_data_p, keep_size, keep, triang, rtn_dim, rtn);CHKERRQ(ierr);
      break;
    default: // shouldn't happen, but give ierr some (nonzero) value for consistency
      ierr = 1;
      break;
  }
  return ierr;
}

#define LEFT_SUBSPACE Full
  #define RIGHT_SUBSPACE Full
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Parity
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Auto
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE

#define LEFT_SUBSPACE Parity
  #define RIGHT_SUBSPACE Full
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Parity
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Auto
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE

#define LEFT_SUBSPACE Auto
  #define RIGHT_SUBSPACE Full
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Parity
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Auto
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE

/*
 * Build the matrix using the appropriate BuildMat function for the subspaces.
 */
PetscErrorCode BuildMat(const msc_t *msc, subspaces_t *subspaces, shell_impl shell, Mat *A)
{
  PetscErrorCode ierr = 0;
  switch (subspaces->left_type) {
    case FULL:
      switch (subspaces->right_type) {
        case FULL:
          ierr = BuildMat_Full_Full(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;

        case PARITY:
          ierr = BuildMat_Full_Parity(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;

        case AUTO:
          ierr = BuildMat_Full_Auto(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;
      }
      break;

    case PARITY:
      switch (subspaces->right_type) {
        case FULL:
          ierr = BuildMat_Parity_Full(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;

        case PARITY:
          ierr = BuildMat_Parity_Parity(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;

        case AUTO:
          ierr = BuildMat_Parity_Auto(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;
      }
      break;

    case AUTO:
      switch (subspaces->right_type) {
        case FULL:
          ierr = BuildMat_Auto_Full(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;

        case PARITY:
          ierr = BuildMat_Auto_Parity(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;

        case AUTO:
          ierr = BuildMat_Auto_Auto(msc, subspaces->left_data, subspaces->right_data, shell, A);
          break;
      }
      break;
  }
  return ierr;
}
