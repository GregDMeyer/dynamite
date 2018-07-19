
include ${SLEPC_DIR}/lib/slepc/conf/slepc_variables
include ${SLEPC_DIR}/lib/slepc/conf/slepc_rules

CFLAGS  = -I..

all: test_shell_kernel

test_shell_kernel: test_shell_kernel.o ../bpetsc_impl.h ../buildmat_template.h ../buildmat_template.c chkopts
	-${CLINKER} -o $@ $@.o ../bpetsc_impl.o ${PETSC_MAT_LIB}
	${RM} test_shell_kernel.o
