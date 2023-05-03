
#include "bcuda_template_shared.h"

#ifdef __cplusplus
extern "C" {
#endif

PetscErrorCode C(BuildGPUShell,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const C(data,LEFT_SUBSPACE)* left_subspace_data,
  const C(data,RIGHT_SUBSPACE)* right_subspace_data,
  int xparity,
  Mat *A);

#ifdef __cplusplus
}
#endif
