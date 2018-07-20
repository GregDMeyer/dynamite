#include "bpetsc_impl.h"

#define LEFT_SUBSPACE Full
  #define RIGHT_SUBSPACE Full
    #include "bpetsc_template_2.c"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE
