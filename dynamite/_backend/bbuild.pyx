
import numpy as np

include "config.pxi"

def get_build_version():
    return DNM_VERSION

def get_build_branch():
    return DNM_BRANCH

def have_gpu_shell():
    return bool(USE_CUDA)

cdef extern from "petsc.h":
    ctypedef int PetscInt

if sizeof(PetscInt) == 4:
    dnm_int_t = np.int32
elif sizeof(PetscInt) == 8:
    dnm_int_t = np.int64
else:
    raise TypeError('Only 32 or 64 bit integers supported.')
