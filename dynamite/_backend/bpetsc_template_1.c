/*
 * This file defines a template for functions that require only one subspace. It should be included
 * multiple times in bpetsc_impl.c, with SUBSPACE defined as the desired value.
 */

#include "bpetsc_template_1.h"

/*
 * this function is actually the same for all subspaces but
 * I keep it here for organizational purposes
 */
static inline PetscInt C(reduce_state,SUBSPACE)(
  PetscInt state,
  const PetscInt* keep,
  PetscInt keep_size
){
  PetscInt rtn = 0;
  PetscInt i;
  for (i=keep_size; i>0; --i) {
    rtn <<= 1;
    rtn |= (state >> keep[i-1]) & 1;
  }
  return rtn;
}

static inline PetscInt C(combine_states,SUBSPACE)(
  PetscInt keep_state,
  PetscInt tr_state,
  const PetscInt* keep,
  PetscInt keep_size,
  PetscInt L
){
  PetscInt rtn = 0;
  PetscInt state_idx, keep_idx, tr_idx;
  PetscInt bit;
  // keep needs to be ordered for this!
  // ordering is checked in rdm below
  keep_idx = 0;
  tr_idx = 0;
  for (state_idx=0; state_idx<L; ++state_idx) {
    if (keep_idx < keep_size && state_idx == keep[keep_idx]) {
      bit = (keep_state >> keep_idx) & 1;
      ++keep_idx;
    }
    else {
      bit = (tr_state >> tr_idx) & 1;
      ++tr_idx;
    }
    rtn |= bit << state_idx;
  }
  return rtn;
}

void C(fill_combine_array,SUBSPACE)(
  PetscInt tr_state,
  PetscInt keep_size,
  const C(data,SUBSPACE)* sub_data_p,
  const PetscInt* keep,
  const PetscScalar *x_array,
  PetscInt *state_array,
  PetscScalar *combine_array,
  PetscInt* n_filled_p)
{
  PetscInt keep_dim, keep_state, full_state;
  PetscInt idx;

  keep_dim = 1<<keep_size;
  *n_filled_p = 0;
  for (keep_state=0; keep_state<keep_dim; ++keep_state) {
    full_state = C(combine_states,SUBSPACE)(keep_state, tr_state, keep, keep_size, sub_data_p->L);
    idx = C(S2I,SUBSPACE)(full_state, sub_data_p);
    if (idx != -1) {
      state_array[*n_filled_p] = keep_state;
      combine_array[*n_filled_p] = x_array[idx];
      ++(*n_filled_p);
    }
  }
}

#undef  __FUNCT__
#define __FUNCT__ "rdm"
PetscErrorCode C(rdm,SUBSPACE)(
  Vec vec,
  const C(data,SUBSPACE)* sub_data_p,
  PetscInt keep_size,
  const PetscInt* keep,
  PetscBool triang,
  PetscInt rtn_dim,
  PetscScalar* rtn
){

  PetscErrorCode ierr;
  const PetscScalar *v0_array;
  PetscInt i, j, n_filled, offset;
  PetscInt tr_state, tr_dim;
  int mpi_size, mpi_rank;
  PetscScalar a;
  Vec v0;
  VecScatter scat;

  PetscInt *state_array;
  PetscScalar *combine_array;

  for (i=1; i<keep_size; ++i) {
    if (keep[i] <= keep[i-1]) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "keep array must be strictly increasing");
    }
  }

  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

  /* scatter to process 0 */
  /* in the future, perhaps will do this in parallel */
  /* could be achieved with a round-robin type deal, like we do with MatMult */
  if (mpi_size > 1) {
    ierr = VecScatterCreateToZero(vec, &scat, &v0);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat, vec, v0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat, vec, v0, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat);CHKERRQ(ierr);
  }
  else {
    v0 = vec;
  }

  /* we're done if we're rank > 0 */
  if (mpi_rank > 0) {
    VecDestroy(&v0);
    return ierr;
  }

  ierr = VecGetArrayRead(v0, &v0_array);CHKERRQ(ierr);

  ierr = PetscMalloc1(1<<keep_size, &state_array);CHKERRQ(ierr);
  ierr = PetscMalloc1(1<<keep_size, &combine_array);CHKERRQ(ierr);

  PetscMemzero(rtn, sizeof(PetscScalar)*rtn_dim*rtn_dim);

  tr_dim = 1 << (sub_data_p->L - keep_size);
  for (tr_state = 0; tr_state < tr_dim; ++tr_state) {
    C(fill_combine_array,SUBSPACE)(tr_state, keep_size, sub_data_p, keep,
      v0_array, state_array, combine_array, &n_filled);
    for (i=0; i<n_filled; ++i) {
      offset = state_array[i]*rtn_dim;
      a = combine_array[i];
      for (j=0; j<n_filled; ++j) {
        rtn[offset + state_array[j]] += a*PetscConj(combine_array[j]);
      }
    }
  }

  ierr = PetscFree(state_array);CHKERRQ(ierr);
  ierr = PetscFree(combine_array);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(v0, &v0_array);CHKERRQ(ierr);
  if (mpi_size > 1) {
    VecDestroy(&v0);
  }

  return ierr;
}
