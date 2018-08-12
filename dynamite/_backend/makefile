
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

# TODO: need to update dependencies

bpetsc_impl.o: bpetsc_impl.c bpetsc_impl.h chkopts

bcuda_impl.o: bcuda_impl.cu bcuda_impl.h chkopts

template_test.o: template_test.c bpetsc_impl.h chkopts

cuda_template_test.o: cuda_template_test.cu bcuda_impl.h chkopts
