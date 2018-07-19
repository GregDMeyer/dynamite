
from . cimport bsubspace

from petsc4py.PETSc cimport Vec,  PetscVec
from petsc4py.PETSc cimport Mat,  PetscMat
from petsc4py.PETSc cimport Scatter, PetscScatter

from petsc4py.PETSc import Error

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

    int ReducedDensityMatrix(PetscInt L,
                             PetscVec x,
                             PetscInt cut_size,
                             bint fillall,
                             np.complex128_t* m)

    int PetscMemoryGetCurrentUsage(PetscLogDouble* mem)
    int PetscMallocGetCurrentUsage(PetscLogDouble* mem)
    int PetscMemorySetGetMaximumUsage()
    int PetscMemoryGetMaximumUsage(PetscLogDouble* mem)
    int PetscMallocGetMaximumUsage(PetscLogDouble* mem)

include "config.pxi"
IF USE_CUDA:
    cdef extern from "cuda_shell.h":
        int BuildMat_CUDAShell(PetscInt L,
                               np.int_t nterms,
                               PetscInt* masks,
                               PetscInt* signs,
                               np.complex128_t* coeffs,
                               PetscMat *A)

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

    # TODO: use an enum for shell types
    if shell == GPU_SHELL:
        IF USE_CUDA:
            if not (left_type == _FULL and right_type == _FULL):
                raise TypeError('Subspaces not currently supported for CUDA shell matrices.')
            ierr = BuildMat_CUDAShell(L,n,&masks[0],&signs[0],&coeffs[0],&M.mat)
        ELSE:
            raise RuntimeError("dynamite was not built with CUDA shell "
                               "functionality (requires nvcc during build).")

    # elif shell == CPU_SHELL:
    #     if left_type == _AUTO or right_type == _AUTO:
    #         raise TypeError('Shell matrices currently not supported for Auto subspace.')

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

@cython.boundscheck(False)
@cython.wraparound(False)
def reduced_density_matrix(Vec v,int cut_size,bint fillall=True):

    # cut_size: number of spins to include in reduced system
    # currently, those will be spins 0 to cut_size-1
    cdef int red_size,ierr,L
    cdef np.ndarray[np.complex128_t,ndim=2] reduced
    cdef Vec v0
    cdef Scatter sc

    # collect to process 0
    sc,v0 = Scatter.toZero(v)
    sc.begin(v,v0)
    sc.end(v,v0)

    # this function will always return None
    # on all processes other than 0
    if v0.getSize() == 0:
        return None

    # get L from the vector's size
    L = v0.getSize().bit_length() - 1

    red_size = 2**cut_size
    reduced = np.zeros((red_size,red_size),dtype=np.complex128,order='C')

    # note: eigvalsh only uses one triangle of the matrix, so allow
    # to only fill half of it
    ierr = ReducedDensityMatrix(L,v0.vec,cut_size,fillall,&reduced[0,0])
    if ierr != 0:
        raise Error(ierr)

    return reduced
