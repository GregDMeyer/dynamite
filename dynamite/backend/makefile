
default: backend_impl.o cuda_shell.o

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

backend_impl.o: backend_impl.c backend_impl.h chkopts

cuda_shell.o: cuda_shell.cu cuda_shell.h chkopts