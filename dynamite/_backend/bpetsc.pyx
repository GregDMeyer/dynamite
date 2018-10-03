
from . cimport bsubspace

from petsc4py.PETSc cimport Vec,  PetscVec
from petsc4py.PETSc cimport Mat,  PetscMat
from petsc4py.PETSc cimport Scatter, PetscScatter

from petsc4py.PETSc import Error, COMM_WORLD

import numpy as np
cimport numpy as np

import cython

cdef extern from "bsubspace_impl.h":
    ctypedef struct subspaces_t:
        int left_type
        int right_type
        void *left_data
        void *right_data

    ctypedef enum subspace_type:
        _FULL "FULL"
        _PARITY "PARITY"
        _AUTO "AUTO"

cdef extern from "bpetsc_impl.h":

    ctypedef int PetscInt
    ctypedef float PetscLogDouble

    ctypedef enum shell_impl:
        NO_SHELL
        CPU_SHELL
        GPU_SHELL

    ctypedef struct msc_t:
      int nmasks
      int* masks
      int* mask_offsets
      int* signs
      np.complex128_t* coeffs

    int BuildMat(msc_t *msc,
                 subspaces_t *subspaces,
                 shell_impl shell,
                 PetscMat *A)

    int ReducedDensityMatrix(
        PetscVec vec,
        int sub_type,
        void* sub_data_p,
        int keep_size,
        int* keep,
        bint triang,
        PetscInt rtn_dim,
        np.complex128_t* rtn_array)

    int PetscMemoryGetCurrentUsage(PetscLogDouble* mem)
    int PetscMallocGetCurrentUsage(PetscLogDouble* mem)
    int PetscMemorySetGetMaximumUsage()
    int PetscMemoryGetMaximumUsage(PetscLogDouble* mem)
    int PetscMallocGetMaximumUsage(PetscLogDouble* mem)

include "config.pxi"

shell_impl_d = {
    False : NO_SHELL,
    'cpu' : CPU_SHELL,
    'gpu' : GPU_SHELL
}

def build_mat(int L,
              PetscInt [:] masks,
              PetscInt [:] mask_offsets,
              PetscInt [:] signs,
              np.complex128_t [:] coeffs,
              subspace_type left_type,
              left_data,
              subspace_type right_type,
              right_data,
              shell_impl shell):

    cdef int ierr, nterms, nmasks
    cdef subspaces_t subspaces
    cdef msc_t msc

    msc.nmasks      = masks.size
    msc.masks       = &masks[0]
    msc.mask_offsets = &mask_offsets[0]
    msc.signs       = &signs[0]
    msc.coeffs      = &coeffs[0]

    M = Mat()

    subspaces.left_type = left_type
    bsubspace.set_data_pointer(left_type, left_data, &(subspaces.left_data))
    subspaces.right_type = right_type
    bsubspace.set_data_pointer(right_type, right_data, &(subspaces.right_data))

    if shell == GPU_SHELL:
        IF not USE_CUDA:
            raise RuntimeError("dynamite was not built with CUDA shell "
                               "functionality (requires nvcc during build).")

    ierr = BuildMat(&msc, &subspaces, shell, &M.mat)

    if ierr != 0:
        raise Error(ierr)

    return M

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

def reduced_density_matrix(Vec v, subspace_type sub_type, sub_data, PetscInt [:] keep, bint triang=True):

    if COMM_WORLD.rank == 0:
        rtn_np = np.zeros((2**keep.size, 2**keep.size), dtype=np.complex128, order='C')
    else:
        # dummy array for the other processes
        rtn_np = np.array([[-1]], dtype=np.complex128)

    cdef np.complex128_t [:,:] rtn = rtn_np
    cdef int ierr
    cdef void* sub_data_p

    bsubspace.set_data_pointer(sub_type, sub_data, &sub_data_p)

    ierr = ReducedDensityMatrix(v.vec, sub_type, sub_data_p, keep.size, &keep[0], triang, rtn.shape[0], &rtn[0,0])
    if ierr != 0:
        raise Error(ierr)

    return rtn_np
