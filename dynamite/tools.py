
from . import config
config.initialize()

import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc

from os import urandom

from .backend import backend

__all__ = [
    'vectonumpy',
    'track_memory',
    'get_max_memory_usage',
    'get_cur_memory_usage']

def vectonumpy(v,toall=False):
    '''
    Collect PETSc vector v (split across processes) to a
    numpy vector on process 0, or to all processes if
    `toall == True`.

    Parameters
    ----------
    v : petsc4py.PETSc.Vec
        The input vector

    toall : bool, optional
        Whether to create numpy vectors on all processes, or
        just on process 0.

    Returns
    -------
    numpy.ndarray or None
        A numpy array of the vector, or ``None``
        on all processes other than 0 if `toall == False`.
    '''

    # collect to process 0
    if toall:
        sc,v0 = PETSc.Scatter.toAll(v)
    else:
        sc,v0 = PETSc.Scatter.toZero(v)
    sc.begin(v,v0)
    sc.end(v,v0)

    # all processes other than 0
    if not toall and v0.getSize() == 0:
        return None

    ret = np.ndarray((v0.getSize(),),dtype=np.complex128)
    ret[:] = v0[:]

    return ret

def track_memory():
    '''
    Begin tracking memory usage for a later call to :meth:`get_max_memory_usage`.
    '''
    return backend.track_memory()

def get_max_memory_usage(which='all'):
    '''
    Get the maximum memory usage up to this point. Only updated whenever
    objects are destroyed (i.e. with :meth:`dynamite.operators.Operator.destroy_mat`)

    .. note::
        :meth:`track_memory` must be called before this function is called,
        and the option ``'-malloc'`` must be supplied to PETSc at runtime to track
        PETSc memory allocations

    Parameters
    ----------
    which : str
        ``'all'`` to return all memory usage for the process, ``'petsc'`` to return
        only memory allocated by PETSc.

    Returns
    -------
    float
        The max memory usage in bytes
    '''
    return backend.get_max_memory_usage(which=which)

def get_cur_memory_usage(which='all'):
    '''
    Get the current memory usage (resident set size) in bytes.

    Parameters
    ----------
    type : str
        ``'all'`` to return all memory usage for the process, ``'petsc'`` to return
        only memory allocated by PETSc.

    Returns
    -------
    float
        The max memory usage in bytes
    '''
    return backend.get_cur_memory_usage(which=which)
