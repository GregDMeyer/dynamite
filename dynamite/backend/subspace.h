
#pragma once
#include <petsc.h>

/* define a struct & enum to hold subspace information */
typedef enum _subspace_type
{
  FULL,
  PARITY
} subspace_type;

typedef struct _Subspaces
{
  subspace_type left_type;
  subspace_type right_type;
  int left_space;
  int right_space;
} Subspaces;

/* these macros assume that values given are valid! */
#define PARITY_MASK(L) (~((-1)<<(L-1)))
#define PARITY_BIT(L) (1<<(L-1))
#define PARITY_StoI(x,p,L) (x & PARITY_MASK(L))
#define PARITY_ItoS(x,p,L) (x|((p^(__builtin_popcount(x)&1))<<(L-1)))

static inline PetscInt get_dimension(PetscInt L,subspace_type type,int space) {
    PetscInt r = -1; /* if a bad type is passed, will return -1 */
    switch (type) {
        case FULL:
            r = 1<<L;
            break;
        case PARITY:
            r = 1<<(L-1);
            break;
    }
    return r;
}
