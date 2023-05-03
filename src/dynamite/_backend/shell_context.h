
#include <petsc.h>

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
  PetscReal *diag;
  subspace_type left_subspace_type;
  void *left_subspace_data;
  subspace_type right_subspace_type;
  void *right_subspace_data;
  PetscReal nrm;
  PetscBool gpu;
} shell_context;
