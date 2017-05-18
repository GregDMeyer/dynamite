from petsc4py.PETSc cimport Vec,  PetscVec
from petsc4py.PETSc cimport Mat,  PetscMat

from petsc4py.PETSc import Error

import numpy as np
cimport numpy as np

import cython

cdef extern from "backend_impl.h":
    ctypedef int PetscInt
    ctypedef float PetscLogDouble

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

    int PetscMemoryGetCurrentUsage(PetscLogDouble* mem)
    int PetscMallocGetCurrentUsage(PetscLogDouble* mem)
    int PetscMemorySetGetMaximumUsage()
    int PetscMemoryGetMaximumUsage(PetscLogDouble* mem)
    int PetscMallocGetMaximumUsage(PetscLogDouble* mem)

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
    cdef int ierr
    ierr = DestroyContext(A.mat)
    if ierr != 0:
        raise Error(ierr)

if sizeof(PetscInt) == 4:
    int_dt = np.int32
elif sizeof(PetscInt) == 8:
    int_dt = np.int64
else:
    raise TypeError('Only 32 or 64 bit integers supported.')

MSC_dtype = np.dtype([('masks',int_dt),('signs',int_dt),('coeffs',np.complex128)])

def track_memory():
    '''
    Begin tracking memory usage for a later call to :meth:`get_max_memory_usage`.
    '''
    cdef int ierr
    ierr = PetscMemorySetGetMaximumUsage()
    if ierr != 0:
        raise Error(ierr)

def get_max_memory_usage(which='all'):
    '''
    Get the maximum memory usage up to this point. Only updated whenever
    objects are destroyed (i.e. with :meth:`dynamite.operators.Operator.destroy_mat`)

    ..note ::
        :meth:`track_memory` must be called before this function is called,
        and the option `'-malloc'` must be supplied to PETSc at runtime to track
        PETSc memory allocations

    Parameters
    ----------
    which : str
        `'all'` to return all memory usage for the process, `'petsc'` to return
        only memory allocated by PETSc.

    Returns
    -------
    float
        The max memory usage in bytes
    '''
    cdef int ierr
    cdef PetscLogDouble mem

    if which == 'all':
        ierr = PetscMemoryGetMaximumUsage(&mem)
    elif which == 'petsc':
        ierr = PetscMallocGetMaximumUsage(&mem)
    else:
        raise ValueError("argument 'which' must be 'all' or 'petsc'")

    if ierr != 0:
        raise Error(ierr)
    return mem

def get_cur_memory_usage(which='all'):
    '''
    Get the current memory usage (resident set size) in bytes.

    Parameters
    ----------
    type : str
        'all' to return all memory usage for the process, 'petsc' to return
        only memory allocated by PETSc.

    Returns
    -------
    float
        The max memory usage in bytes
    '''
    cdef int ierr
    cdef PetscLogDouble mem

    if which == 'all':
        ierr = PetscMemoryGetCurrentUsage(&mem)
    elif which == 'petsc':
        ierr = PetscMallocGetCurrentUsage(&mem)
    else:
        raise ValueError("argument 'which' must be 'all' or 'petsc'")

    if ierr != 0:
        raise Error(ierr)
    return mem

cdef packed struct MSC_t:
    PetscInt masks
    PetscInt signs
    np.complex128_t coeffs

def product_of_terms(np.ndarray[MSC_t,ndim=1] factors):
    cdef MSC_t factor,prod
    cdef np.ndarray[MSC_t,ndim=1] prod_array
    cdef PetscInt flipped
    cdef int flip

    prod = factors[0]

    for factor in factors[1:]:

        # keep the sign correct after spin flips.
        # this is crucial... otherwise everything
        # would commute!
        flipped = prod.masks & factor.signs
        flip = 1
        while flipped:
            flip *= -1
            flipped = flipped & (flipped-1)

        prod.masks = prod.masks ^ factor.masks
        prod.signs = prod.signs ^ factor.signs

        prod.coeffs *= (factor.coeffs * flip)

    prod_array = np.ndarray((1,),dtype=MSC_dtype)
    prod_array[0] = prod

    return prod_array