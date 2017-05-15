from petsc4py.PETSc cimport Vec,  PetscVec
from petsc4py.PETSc cimport Mat,  PetscMat

from petsc4py.PETSc import Error

import numpy as np
cimport numpy as np

import cython

cdef extern from "backend_impl.h":
    ctypedef int PetscInt

    int BuildMat_Full(PetscInt L,
                      np.int_t nterms,
                      PetscInt* masks,
                      PetscInt* signs,
                      np.complex128_t* coeffs,
                      PetscMat *A)

    int BuildMat_Shell(PetscInt L,
                       np.int_t nterms,
                       PetscInt* masks,
                       PetscInt* signs,
                       np.complex128_t* coeffs,
                       PetscMat *A)

    int DestroyContext(PetscMat A)

@cython.boundscheck(False)
@cython.wraparound(False)
def build_mat(int L,
              np.ndarray[PetscInt,mode="c"] masks not None,
              np.ndarray[PetscInt,mode="c"] signs not None,
              np.ndarray[np.complex128_t,mode="c"] coeffs not None,
              bint shell=False):

    cdef int ierr,n

    M = Mat()
    n = masks.shape[0]

    if shell:
        ierr = BuildMat_Shell(L,n,&masks[0],&signs[0],&coeffs[0],&M.mat)
    else:
        ierr = BuildMat_Full(L,n,&masks[0],&signs[0],&coeffs[0],&M.mat)

    if ierr != 0:
        raise Error(ierr)

    return M

def destroy_shell_context(Mat A):
    DestroyContext(A.mat)

if sizeof(PetscInt) == 4:
    int_dt = np.int32
elif sizeof(PetscInt) == 8:
    int_dt = np.int64
else:
    raise TypeError('Only 32 or 64 bit integers supported.')

def MSC_dtype():
    return np.dtype([('masks',int_dt),('signs',int_dt),('coeffs',np.complex128)])
