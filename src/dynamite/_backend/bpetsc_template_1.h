
#define CONCAT_U(a, b) a ## _ ## b
#define C(a, b) CONCAT_U(a, b)

/*
 * This function is called to build any matrix.
 */
PetscErrorCode C(rdm,SUBSPACE)(
  Vec vec,
  const C(data,SUBSPACE)* sub_data_p,
  PetscInt keep_size,
  const PetscInt* keep,
  PetscBool triang,
  PetscInt rtn_dim,
  PetscScalar* rtn
);
