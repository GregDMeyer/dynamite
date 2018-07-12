#include "bpetsc_impl.h"

#define LEFT_SUBSPACE Full
  #define RIGHT_SUBSPACE Full
    #include "buildmat_template.c"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE
