
#pragma once
#include <petscsys.h>

typedef struct _shell_context {
  PetscInt L;
  PetscInt nterms;
  PetscInt *state_map;
  PetscInt *choose_array;
  PetscInt *masks;
  PetscInt *signs;
  PetscScalar *coeffs;
  PetscReal nrm;
  PetscBool gpu;
} shell_context;
