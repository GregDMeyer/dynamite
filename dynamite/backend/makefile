# -*- makefile -*-

MPIEXEC=

.PHONY:test
test: run clean

SCRIPT=run_demo
MODULE=backend
ERROR_SUPPRESS=-Wno-cpp -Wno-strict-prototypes

.PHONY:build
build: ${MODULE}.so

.PHONY:run
run: build
	${MPIEXEC} ${PYTHON} ${SCRIPT}.py

${MODULE}.so: ${MODULE}.pyx ${MODULE}_impl.h ${MODULE}_impl.c
	CC=${CC} F90=${FC} CFLAGS="${ERROR_SUPPRESS}" \
	${PYTHON} setup.py -q build_ext --inplace
	${RM} -r build ${MODULE}.c

.PHONY:clean
clean::
	${RM} ${MODULE}.c ${MODULE}*.so
	${RM} *.py[co]
	${RM} -r __pycache__

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
MPIEXEC=

PYTHON=python3
