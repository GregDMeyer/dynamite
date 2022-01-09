# cython: language_level=3

import numpy as np

include "config.pxi"

def get_build_version():
    return DNM_VERSION

def get_build_branch():
    return DNM_BRANCH

cdef extern from "petsc.h":
    ctypedef int PetscInt
    ctypedef int PetscBool
    int PetscInitialized(PetscBool *isInitialized)

cdef extern from "bpetsc_impl.h":
    int DNM_PETSC_COMPLEX

def have_gpu_shell():
    return bool(USE_CUDA)

if sizeof(PetscInt) == 4:
    dnm_int_t = np.int32
elif sizeof(PetscInt) == 8:
    dnm_int_t = np.int64
else:
    raise TypeError('Only 32 or 64 bit integers supported.')

def complex_enabled():
    return bool(DNM_PETSC_COMPLEX)

def petsc_initialized():
    cdef PetscBool rtn
    cdef int ierr
    ierr = PetscInitialized(&rtn)
    if ierr != 0:
        raise RuntimeError('Error while checking for PETSc initialization.')
    return bool(rtn)
