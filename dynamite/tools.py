
from . import config
config.initialize()

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from .backend import backend

__all__ = [
    'get_version',
    'get_version_str',
    'vectonumpy',
    'track_memory',
    'get_max_memory_usage',
    'get_cur_memory_usage']

def get_version():
    '''
    Gets the version information for dynamite, and the PETSc and SLEPc libraries it's built on.

    Returns
    -------

    dict
        A dictionary with the keys 'PETSc', 'SLEPc', and 'dynamite', each of which contains version
        information for the respective library.
    '''

    rtn = {}
    rtn['PETSc'] = PETSc.Sys.getVersionInfo()
    rtn['SLEPc'] = SLEPc.Sys.getVersionInfo()
    rtn['dynamite'] = {}
    rtn['dynamite']['commit'] = backend.get_build_version()
    rtn['dynamite']['branch'] = backend.get_build_branch()
    return rtn

def get_version_str():
    '''
    Get a string with the version information for PETSc, SLEPc, and dynamite.

    Returns
    -------

    str
        The version string
    '''

    info = get_version()

    rtn = 'dynamite commit {dnm_v} on branch "{dnm_branch}" ' +\
          'built with PETSc {PETSc_v} and SLEPc {SLEPc_v}'
    rtn = rtn.format(
        dnm_v = info['dynamite']['commit'],
        dnm_branch = info['dynamite']['branch'],
        PETSc_v = '.'.join([str(info['PETSc'][k]) for k in ['major', 'minor', 'subminor']]),
        SLEPc_v = '.'.join([str(info['SLEPc'][k]) for k in ['major', 'minor', 'subminor']]),
    )

    return rtn

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
