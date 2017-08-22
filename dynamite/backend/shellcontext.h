
#pragma once
#include <petscsys.h>

typedef struct _shell_context {
  PetscInt L;
  PetscInt nterms;
  PetscInt *masks;
  PetscInt *signs;
  PetscScalar *coeffs;
  PetscReal nrm;
  PetscBool gpu;
} shell_context;
