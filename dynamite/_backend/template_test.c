#include "bpetsc_impl.h"

#define SUBSPACE Full
  #include "bpetsc_template_1.c"
#undef SUBSPACE

#define SUBSPACE Parity
  #include "bpetsc_template_1.c"
#undef SUBSPACE

#define LEFT_SUBSPACE Full
  #define RIGHT_SUBSPACE Full
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE

#define LEFT_SUBSPACE Parity
  #define RIGHT_SUBSPACE Parity
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE
