
from . import config
config.initialize()

import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc

from os import urandom

from .backend import backend

__all__ = [
    'build_state',
    'vectonumpy',
    'track_memory',
    'get_max_memory_usage',
    'get_cur_memory_usage']

def build_state(L = None,state = 0,seed = None):
    '''
    Build a PETSc vector representing a state.

    .. note::
        State indices go from right-to-left. For example,
        the state "UUUUD" has the spin at index 0 down
        and all the rest of the spins up.

    Parameters
    ----------
    L : int
        The length of the spin chain. Can be omitted if a global
        L has been set with :meth:`dynamite.Config.global_L`.

    state : int or str, optional
        The desired state. Can either be an integer whose
        binary representation represents the spin configuration
        (0=↓, 1=↑) of a product state, or a string of the form
        ``"DUDDU...UDU"`` (D=↓, U=↑). If it is a string, the string's
        length must equal ``L``. One can also pass ``state='random'``
        to generate a random, normalized state (not a product state,
        random values for all components).

    seed : int, optional
        The seed for the random number generator, when generating
        a random state. Has no effect without the option ``state='random'``.
        Note that on multiple processes, the seed is incremented by the process
        number, to prevent different parts of the vector from having the
        same random values.

    Returns
    -------
    petsc4py.PETSc.Vec
        The product state
    '''

    if L is None:
        L = config.global_L

    v = PETSc.Vec().create()
    v.setSizes(1<<L)
    v.setFromOptions()

    if state == 'random':
        istart,iend = v.getOwnershipRange()

        R = np.random.RandomState()

        if seed is None:
            try:
                seed = int.from_bytes(urandom(4),'big',signed=False)
            except NotImplementedError:
                raise RuntimeError('Could not access urandom for random number '
                                   'initialization. Please manually set a seed.')

        R.seed(seed + PETSc.COMM_WORLD.rank)

        local_size = iend-istart
        v[istart:iend] = R.standard_normal(local_size) + \
                            1j*R.standard_normal(local_size)

    else:
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

    if state == 'random':
        v.normalize()

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
