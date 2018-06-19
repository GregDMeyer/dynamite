
#pragma once

#include <petsc.h>
typedef PetscInt dnm_cint_t;

/* define a struct & enum to hold subspace information */
typedef enum _subspace_type
{
  FULL,
  PARITY,
  AUTO
} subspace_type;

typedef struct _Subspaces
{
  subspace_type left_type;
  subspace_type right_type;
  int left_space;
  int right_space;
} Subspaces;

/***** PARITY *****/

#define PARITY_MASK(L) (~((-1)<<((L)-1)))
#define PARITY_BIT(L) (1<<((L)-1))

#define PARITY_S2I(x,p,L) ((x) & PARITY_MASK(L))
#define PARITY_I2S(x,p,L) ((x)|(((p)^(__builtin_popcount(x)&1))<<((L)-1)))

static inline dnm_cint_t Parity_Dim(int L, int space) {
  return 1 << (L-1);
}

static inline void Parity_I2S(const dnm_cint_t* idxs, int n, int L, int space, dnm_cint_t* states) {
  for (dnm_cint_t i = 0; i < n; ++i) {
    states[i] = PARITY_I2S(idxs[i], space, L);
  }
}

static inline void Parity_S2I(const dnm_cint_t* states, int n, int L, int space, dnm_cint_t* idxs) {
  /* parity s2i doesn't actually need to know which space it's in */
  for (dnm_cint_t i = 0; i < n; ++i) {
    idxs[i] = PARITY_S2I(states[i], 0, L);
  }
}


/***** AUTO *****/

// TODO

dnm_cint_t Auto_Dim(int L, int space);
void Auto_I2S(dnm_cint_t* idxs, int n, int L, int space, dnm_cint_t* states);
void Auto_S2I(dnm_cint_t* states, int n, int L, int space, dnm_cint_t* idxs);
