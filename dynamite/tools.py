
from . import initialize
initialize()

import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc

from .backend import backend

__all__ = [
    'build_state',
    'vectonumpy',
    'track_memory',
    'get_max_memory_usage',
    'get_cur_memory_usage']

def build_state(L,state = 0):
    '''
    Build a PETSc vector representing some product state.

    .. note::
        State indices go from right-to-left. For example,
        the state "UUUUD" has the spin at index 0 down
        and all the rest of the spins up.

    Parameters
    ----------
    L : int
        The length of the spin chain

    state : int or str, optional
        The product state. Can either be an integer whose
        binary representation represents the spin configuration
        (0=↓, 1=↑) or a string of the form ``"DUDDU...UDU"``
        (D=↓, U=↑). If it is a string, the string's length must
        equal ``L``.

    Returns
    -------
    petsc4py.PETSc.Vec
        The product state
    '''

    v = PETSc.Vec().create()
    v.setSizes(1<<L)
    v.setFromOptions()

    if isinstance(state,str):
        state_str = state
        state = 0
        if len(state_str) != L:
            raise ValueError('state string must have length L')
        if not all(c in ['U','D'] for c in state_str):
            raise ValueError('only character U and D allowed in state')
        for i,c in enumerate(state_str[::-1]):
            if c == 'U':
                state += 1<<i

    elif not isinstance(state,int):
        raise TypeError('State must be an int or str.')

    if not 0 <= state < 2**L:
        raise ValueError('Requested state out of bounds (0,2**L).')

    v[state] = 1

    v.assemblyBegin()
    v.assemblyEnd()

    return v

def vectonumpy(v):
    '''
    Collect PETSc vector v (split across processes) to a
    numpy vector on process 0.

    Parameters
    ----------
    v : petsc4py.PETSc.Vec
        The input vector

    Returns
    -------
    numpy.ndarray or None
        A numpy array of the vector on process 0, ``None``
        on all other processes.
    '''

    # collect to process 0
    sc,v0 = PETSc.Scatter.toZero(v)
    sc.begin(v,v0)
    sc.end(v,v0)

    # all processes other than 0
    if v0.getSize() == 0:
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
