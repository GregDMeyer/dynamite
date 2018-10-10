
#pragma once

#include <petsc.h>

// TODO: include this in a different header?
#ifdef PETSC_USE_64BIT_INDICES
  #define builtin_parity __builtin_parityl
#else
  #define builtin_parity __builtin_parity
#endif

/* define a struct & enum to hold subspace information */
// TODO: more descriptive name?
typedef enum _subspace_type
{
  FULL,
  PARITY,
  AUTO
} subspace_type;

typedef struct _subspaces_t
{
  subspace_type left_type;
  subspace_type right_type;
  void *left_data;
  void *right_data;
} subspaces_t;

/***** FULL *****/

typedef struct _data_Full
{
  PetscInt L;
} data_Full;

static inline PetscErrorCode CopySubspaceData_Full(data_Full** out_p, const data_Full* in) {
  PetscErrorCode ierr;
  ierr = PetscMalloc1(1, out_p);CHKERRQ(ierr);
  ierr = PetscMemcpy(*out_p, in, sizeof(data_Full));CHKERRQ(ierr);
  return ierr;
}

static inline PetscErrorCode DestroySubspaceData_Full(data_Full* data) {
  PetscErrorCode ierr;
  ierr = PetscFree(data);CHKERRQ(ierr);
  return ierr;
}

static inline PetscInt Dim_Full(const data_Full* data) {
  return (PetscInt)1 << data->L;
}

static inline PetscInt S2I_Full(PetscInt state, const data_Full* data) {
  return state;
}

static inline PetscInt S2I_nocheck_Full(PetscInt state, const data_Full* data) {
  return state;
}

static inline PetscInt I2S_Full(PetscInt idx, const data_Full* data) {
  return idx;
}

static inline void S2I_Full_array(int n, const data_Full* data, const PetscInt* states, PetscInt* idxs) {
  PetscMemcpy(idxs, states, n*sizeof(PetscInt));
}

static inline void I2S_Full_array(int n, const data_Full* data, const PetscInt* idxs, PetscInt* states) {
  PetscMemcpy(states, idxs, n*sizeof(PetscInt));
}

/***** PARITY *****/

typedef struct _data_Parity
{
  PetscInt L;
  PetscInt space;
} data_Parity;

static inline PetscErrorCode CopySubspaceData_Parity(data_Parity** out_p, const data_Parity* in) {
  PetscErrorCode ierr;
  ierr = PetscMalloc1(1, out_p);CHKERRQ(ierr);
  ierr = PetscMemcpy(*out_p, in, sizeof(data_Parity));CHKERRQ(ierr);
  return ierr;
}

static inline PetscErrorCode DestroySubspaceData_Parity(data_Parity* data) {
  PetscErrorCode ierr;
  ierr = PetscFree(data);CHKERRQ(ierr);
  return ierr;
}

static inline PetscInt Dim_Parity(const data_Parity* data) {
  return (PetscInt)1 << (data->L-1);
}

static inline PetscInt S2I_Parity(PetscInt state, const data_Parity* data) {
  if (builtin_parity(state) == data->space) {
    return state>>1;
  }
  else {
    return (PetscInt)(-1);
  }
}

static inline PetscInt S2I_nocheck_Parity(PetscInt state, const data_Parity* data) {
  return state>>1;
}

static inline PetscInt I2S_Parity(PetscInt idx, const data_Parity* data) {
  return (idx<<1) | (builtin_parity(idx) ^ data->space);
}

static inline void S2I_Parity_array(int n, const data_Parity* data, const PetscInt* states, PetscInt* idxs) {
  PetscInt i;
  for (i = 0; i < n; ++i) {
    idxs[i] = S2I_Parity(states[i], data);
  }
}

static inline void I2S_Parity_array(int n, const data_Parity* data, const PetscInt* idxs, PetscInt* states) {
  PetscInt i;
  for (i = 0; i < n; ++i) {
    states[i] = I2S_Parity(idxs[i], data);
  }
}

/***** AUTO *****/

typedef struct _data_Auto
{
  PetscInt L;
  PetscInt dim;
  PetscInt* state_map;
  PetscInt* rmap_indices;
  PetscInt* rmap_states;
} data_Auto;

static inline PetscErrorCode CopySubspaceData_Auto(data_Auto** out_p, const data_Auto* in) {
  PetscErrorCode ierr;
  ierr = PetscMalloc1(1, out_p);CHKERRQ(ierr);
  ierr = PetscMemcpy(*out_p, in, sizeof(data_Auto));CHKERRQ(ierr);

  ierr = PetscMalloc1(in->dim, &((*out_p)->state_map));CHKERRQ(ierr);
  ierr = PetscMemcpy((*out_p)->state_map, in->state_map, in->dim*sizeof(PetscInt));CHKERRQ(ierr);

  ierr = PetscMalloc1(in->dim, &((*out_p)->rmap_indices));CHKERRQ(ierr);
  ierr = PetscMemcpy((*out_p)->rmap_indices, in->rmap_indices, in->dim*sizeof(PetscInt));CHKERRQ(ierr);

  ierr = PetscMalloc1(in->dim, &((*out_p)->rmap_states));CHKERRQ(ierr);
  ierr = PetscMemcpy((*out_p)->rmap_states, in->rmap_states, in->dim*sizeof(PetscInt));CHKERRQ(ierr);

  return ierr;
}

static inline PetscErrorCode DestroySubspaceData_Auto(data_Auto* data) {
  PetscErrorCode ierr;
  ierr = PetscFree(data->state_map);CHKERRQ(ierr);
  ierr = PetscFree(data->rmap_indices);CHKERRQ(ierr);
  ierr = PetscFree(data->rmap_states);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);
  return ierr;
}

static inline PetscInt Dim_Auto(const data_Auto* data) {
  return data->dim;
}

static inline PetscInt S2I_Auto(PetscInt state, const data_Auto* data) {
  /* do a binary search on rmap_states */
  PetscInt left, right, mid;
  left = 0;
  right = data->dim;
  while (left <= right) {
    mid = left + (right-left)/2;
    if (data->rmap_states[mid] == state) {
      return data->rmap_indices[mid];
    }

    if (data->rmap_states[mid] < state) {
      left = mid + 1;
    }
    else {
      right = mid - 1;
    }
  }
  /* element was not in the array */
  return -1;
}

static inline PetscInt I2S_Auto(PetscInt idx, const data_Auto* data) {
  return data->state_map[idx];
}

static inline void S2I_Auto_array(int n, const data_Auto* data, const PetscInt* states, PetscInt* idxs) {
  PetscInt i;
  for (i = 0; i < n; ++i) {
    idxs[i] = S2I_Auto(states[i], data);
  }
}

static inline void I2S_Auto_array(int n, const data_Auto* data, const PetscInt* idxs, PetscInt* states) {
  PetscInt i;
  for (i = 0; i < n; ++i) {
    states[i] = I2S_Auto(idxs[i], data);
  }
}
