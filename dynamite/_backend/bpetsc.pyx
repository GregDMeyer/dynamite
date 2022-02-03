# cython: language_level=3

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
        _EXPLICIT "EXPLICIT"

cdef extern from "bpetsc_impl.h":

    ctypedef int PetscInt
    ctypedef float PetscLogDouble

    int DNM_PETSC_COMPLEX

    ctypedef enum shell_impl:
        NO_SHELL
        CPU_SHELL
        GPU_SHELL

    ctypedef struct msc_t:
      int nmasks
      int* masks
      int* mask_offsets
      int* signs
      void* coeffs

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
        void* rtn_array)

    int PetscMemoryGetCurrentUsage(PetscLogDouble* mem)
    int PetscMallocGetCurrentUsage(PetscLogDouble* mem)
    int PetscMemorySetGetMaximumUsage()
    int PetscMemoryGetMaximumUsage(PetscLogDouble* mem)
    int PetscMallocGetMaximumUsage(PetscLogDouble* mem)

include "config.pxi"

def build_mat(int L,
              PetscInt [:] masks,
              PetscInt [:] mask_offsets,
              PetscInt [:] signs,
              np.complex128_t [:] coeffs,
              subspace_type left_type,
              left_data,
              subspace_type right_type,
              right_data,
              bint shell,
              bint gpu):

    cdef int ierr, nterms, nmasks
    cdef subspaces_t subspaces
    cdef msc_t msc
    cdef shell_impl which_shell

    cdef np.float64_t [:] real_coeffs

    msc.nmasks      = masks.size
    msc.masks       = &masks[0]
    msc.mask_offsets = &mask_offsets[0]
    msc.signs       = &signs[0]

    if DNM_PETSC_COMPLEX:
        msc.coeffs = <void*>&coeffs[0]
    else:
        # check that all the coefficients were actually real
        if not np.all(np.isreal(coeffs)):
            raise ValueError('matrix has complex entries but PETSc was '
                             'configured for real numbers')

        real_coeffs_np = np.ascontiguousarray(np.real(coeffs), dtype=np.float64)
        real_coeffs = real_coeffs_np

        msc.coeffs = <void*>&real_coeffs[0]

    M = Mat()

    subspaces.left_type = left_type
    bsubspace.set_data_pointer(left_type, left_data, &(subspaces.left_data))
    subspaces.right_type = right_type
    bsubspace.set_data_pointer(right_type, right_data, &(subspaces.right_data))

    if gpu:
        IF not USE_CUDA:
            raise RuntimeError("dynamite was not built with CUDA shell "
                               "functionality (requires nvcc during build).")

    if not shell:
        which_shell = NO_SHELL
    else:
        if gpu:
            which_shell = GPU_SHELL
        else:
            which_shell = CPU_SHELL

    ierr = BuildMat(&msc, &subspaces, which_shell, &M.mat)

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

    if DNM_PETSC_COMPLEX:
        matrix_dtype = np.complex128
    else:
        matrix_dtype = np.float64

    if COMM_WORLD.rank == 0:
        rtn_np = np.zeros((2**keep.size, 2**keep.size), dtype=matrix_dtype, order='C')
    else:
        # dummy array for the other processes
        rtn_np = np.array([[-1]], dtype=matrix_dtype)

    cdef int ierr
    cdef void* sub_data_p

    bsubspace.set_data_pointer(sub_type, sub_data, &sub_data_p)

    cdef np.complex128_t [:,:] rtn_c
    cdef np.float64_t [:,:] rtn_f

    if DNM_PETSC_COMPLEX:
        rtn_c = rtn_np
        ierr = ReducedDensityMatrix(v.vec, sub_type, sub_data_p, keep.size, &keep[0], triang, rtn_c.shape[0], <void*>&rtn_c[0,0])
    else:
        rtn_f = rtn_np
        ierr = ReducedDensityMatrix(v.vec, sub_type, sub_data_p, keep.size, &keep[0], triang, rtn_f.shape[0], <void*>&rtn_f[0,0])

    if ierr != 0:
        raise Error(ierr)

    return rtn_np
