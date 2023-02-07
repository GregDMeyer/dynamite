
#pragma once

#include <petsc.h>

#ifdef PETSC_USE_64BIT_INDICES
  #define builtin_parity __builtin_parityl
  #define builtin_popcount __builtin_popcountl
  #define builtin_ctz __builtin_ctzl
#else
  #define builtin_parity __builtin_parity
  #define builtin_popcount __builtin_popcount
  #define builtin_ctz __builtin_ctz
#endif

/* define a struct & enum to hold subspace information */
typedef enum _subspace_type
{
  FULL,
  PARITY,
  EXPLICIT,
  SPIN_CONSERVE
} subspace_type;

typedef struct _subspaces_t
{
  subspace_type left_type;
  subspace_type right_type;
  void *left_data;
  void *right_data;
} subspaces_t;

/*
 * NOTE: none of the functions in this file handle XParity---that is handled elsewhere.
 * when called on XParity all of these functions give correct values for the *parent*
 * subspace
 */

/***** FULL *****/

typedef struct _data_Full
{
  PetscInt L;
} data_Full;

static inline PetscErrorCode CopySubspaceData_Full(data_Full** out_p, const data_Full* in) {
  PetscCall(PetscMalloc1(1, out_p));
  PetscCall(PetscMemcpy(*out_p, in, sizeof(data_Full)));
  return 0;
}

static inline PetscErrorCode DestroySubspaceData_Full(data_Full* data) {
  PetscCall(PetscFree(data));
  return 0;
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
  PetscAssert(idx >= 0 && idx < Dim_Full(data), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
              "Index %d is out of bounds for subspace of dimension %d.\n",
              idx, Dim_Full(data));
  return idx;
}

static inline PetscInt NextState_Full(
  PetscInt prev_state,
  PetscInt idx,
  const data_Full* data
)
{
  return I2S_Full(idx, data);
};

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
  PetscCall(PetscMalloc1(1, out_p));
  PetscCall(PetscMemcpy(*out_p, in, sizeof(data_Parity)));
  return 0;
}

static inline PetscErrorCode DestroySubspaceData_Parity(data_Parity* data) {
  PetscCall(PetscFree(data));
  return 0;
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
  PetscAssert(idx >= 0 && idx < Dim_Parity(data), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
              "Index %d is out of bounds for subspace of dimension %d.\n",
              idx, Dim_Parity(data));
  return (idx<<1) | (builtin_parity(idx) ^ data->space);
}

static inline PetscInt NextState_Parity(
  PetscInt prev_state,
  PetscInt idx,
  const data_Parity* data
)
{
  return I2S_Parity(idx, data);
};

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

/***** SPIN CONSERVE *****/

typedef struct _data_SpinConserve
{
  PetscInt L;
  PetscInt k;
  PetscInt ld_nchoosek;
  PetscInt* nchoosek;
} data_SpinConserve;

static inline PetscErrorCode CopySubspaceData_SpinConserve(data_SpinConserve** out_p, const data_SpinConserve* in) {
  PetscInt len_nchoosek = (in->k+1)*in->ld_nchoosek;

  PetscCall(PetscMalloc1(1, out_p));
  PetscCall(PetscMemcpy(*out_p, in, sizeof(data_SpinConserve)));

  PetscCall(PetscMalloc1(len_nchoosek, &((*out_p)->nchoosek)));
  PetscCall(PetscMemcpy((*out_p)->nchoosek, in->nchoosek, len_nchoosek*sizeof(PetscInt)));

  return 0;
}

static inline PetscErrorCode DestroySubspaceData_SpinConserve(data_SpinConserve* data) {
  PetscCall(PetscFree(data->nchoosek));
  PetscCall(PetscFree(data));
  return 0;
}

static inline PetscInt Dim_SpinConserve(const data_SpinConserve* data) {
  return data->nchoosek[data->k*data->ld_nchoosek + data->L];
}

static inline PetscInt S2I_nocheck_SpinConserve(PetscInt state, const data_SpinConserve* data) {
  PetscInt n, k=0, idx=0;

  while (state) {
    n = builtin_ctz(state);
    k++;
    if (k <= n) idx += data->nchoosek[k*data->ld_nchoosek + n];
    state &= state-1;  // pop least significant bit off of state
  }

  return idx;
}

static inline PetscInt S2I_SpinConserve(PetscInt state, const data_SpinConserve* data) {
  if (builtin_popcount(state) != data->k) return (PetscInt)(-1);

  return S2I_nocheck_SpinConserve(state, data);
}

static inline PetscInt I2S_SpinConserve(PetscInt idx, const data_SpinConserve* data) {
  PetscAssert(idx >= 0 && idx < Dim_SpinConserve(data), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
              "Index %d is out of bounds for subspace of dimension %d.\n",
              idx, Dim_SpinConserve(data));

  PetscInt state = 0;
  PetscInt k = data->k;
  PetscInt current;
  for (PetscInt n=data->L; n>0; --n) {
    state <<= 1;
    current = (k > n-1) ? 0 : data->nchoosek[k*data->ld_nchoosek + n-1];
    if (idx >= current) {
        idx -= current;
        k--;
        state |= 1 ;
    }
  }
  return state;
}

static inline PetscInt NextState_SpinConserve(
  PetscInt prev_state,
  PetscInt idx,
  const data_SpinConserve* data
)
{
  PetscInt tz = builtin_ctz(prev_state);
  prev_state >>= tz;
  ++prev_state;
  PetscInt to = builtin_ctz(prev_state);
  prev_state >>= to;
  prev_state <<= to + tz;
  prev_state |= (1 << (to - 1)) - 1;

  return prev_state;
};


static inline void S2I_SpinConserve_array(int n, const data_SpinConserve* data, const PetscInt* states, PetscInt* idxs) {
  PetscInt i;

  for (i = 0; i < n; ++i) {
    idxs[i] = S2I_SpinConserve(states[i], data);
  }
}

static inline void I2S_SpinConserve_array(int n, const data_SpinConserve* data, const PetscInt* idxs, PetscInt* states) {
  PetscInt i;
  for (i = 0; i < n; ++i) {
    states[i] = I2S_SpinConserve(idxs[i], data);
  }
}

/***** EXPLICIT *****/

typedef struct _data_Explicit
{
  PetscInt L;
  PetscInt dim;
  PetscInt* state_map;
  PetscInt* rmap_indices;
  PetscInt* rmap_states;
} data_Explicit;

static inline PetscErrorCode CopySubspaceData_Explicit(data_Explicit** out_p, const data_Explicit* in) {
  PetscCall(PetscMalloc1(1, out_p));
  PetscCall(PetscMemcpy(*out_p, in, sizeof(data_Explicit)));

  PetscCall(PetscMalloc1(in->dim, &((*out_p)->state_map)));
  PetscCall(PetscMemcpy((*out_p)->state_map, in->state_map, in->dim*sizeof(PetscInt)));

  if (in->rmap_indices != NULL) {
    PetscCall(PetscMalloc1(in->dim, &((*out_p)->rmap_indices)));
    PetscCall(PetscMemcpy((*out_p)->rmap_indices, in->rmap_indices, in->dim*sizeof(PetscInt)));
  }

  PetscCall(PetscMalloc1(in->dim, &((*out_p)->rmap_states)));
  PetscCall(PetscMemcpy((*out_p)->rmap_states, in->rmap_states, in->dim*sizeof(PetscInt)));

  return 0;
}

static inline PetscErrorCode DestroySubspaceData_Explicit(data_Explicit* data) {
  PetscCall(PetscFree(data->state_map));
  if (data->rmap_indices != NULL) {
    PetscCall(PetscFree(data->rmap_indices));
  }
  PetscCall(PetscFree(data->rmap_states));
  PetscCall(PetscFree(data));
  return 0;
}

static inline PetscInt Dim_Explicit(const data_Explicit* data) {
  return data->dim;
}

static inline PetscInt S2I_Explicit(PetscInt state, const data_Explicit* data) {
  /* do a binary search on rmap_states */
  PetscInt left, right, mid;
  left = 0;
  right = data->dim-1;

  while (left <= right) {
    mid = (left + right)/2;
    if (data->rmap_states[mid] == state) {
      if (data->rmap_indices != NULL) {
        return data->rmap_indices[mid];
      }
      else {
        return mid;
      }
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

static inline PetscInt I2S_Explicit(PetscInt idx, const data_Explicit* data) {
  PetscAssert(idx >= 0 && idx < Dim_Explicit(data), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
              "Index %d is out of bounds for subspace of dimension %d.\n",
              idx, Dim_Explicit(data));
  return data->state_map[idx];
}

static inline PetscInt NextState_Explicit(
  PetscInt prev_state,
  PetscInt idx,
  const data_Explicit* data
)
{
  return I2S_Explicit(idx, data);
};

static inline void S2I_Explicit_array(int n, const data_Explicit* data, const PetscInt* states, PetscInt* idxs) {
  PetscInt i;
  for (i = 0; i < n; ++i) {
    idxs[i] = S2I_Explicit(states[i], data);
  }
}

static inline void I2S_Explicit_array(int n, const data_Explicit* data, const PetscInt* idxs, PetscInt* states) {
  PetscInt i;
  for (i = 0; i < n; ++i) {
    states[i] = I2S_Explicit(idxs[i], data);
  }
}
