
#pragma once
#include <petscsys.h>
#include "bsubspace_impl.h"

typedef struct _shell_context {
  PetscInt L;
  Subspaces s;
  PetscInt nterms;
  PetscInt *masks;
  PetscInt *signs;
  PetscReal *coeffs;
  PetscBool *creal;
  PetscInt **lookup;
  PetscInt *mask_starts;
  PetscReal nrm;
  PetscBool gpu;
} shell_context;
