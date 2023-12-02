'''
Various tools useful for writing and analyzing dynamite programs.
'''

import warnings


def MPI_COMM_WORLD():
    '''
    Returns PETSc's COMM_WORLD object. Can be converted to an mpi4py
    communicator object via the ``.tompi4py()`` method.
    '''
    from . import config
    config._initialize()  # dynamite must be initialized before importing PETSc
    from petsc4py import PETSc
    return PETSc.COMM_WORLD


def mpi_print(*args, rank=0, **kwargs):
    '''
    Print from only a single MPI rank, default rank 0.

    Aside from the extra "rank" keywork argument, call signature is the same
    as Python 3's ``print()`` function.
    '''
    if MPI_COMM_WORLD().rank == rank:
        print(*args, **kwargs)


def get_version():
    '''
    Gets the version information for dynamite, and the PETSc and SLEPc libraries it's built on.

    Returns
    -------

    dict
        A dictionary with the keys 'PETSc', 'SLEPc', and 'dynamite'
    '''

    from ._backend import bbuild

    from . import config
    config._initialize()
    from petsc4py import PETSc
    from slepc4py import SLEPc

    rtn = {}
    rtn['PETSc'] = PETSc.Sys.getVersionInfo()
    rtn['SLEPc'] = SLEPc.Sys.getVersionInfo()
    rtn['dynamite'] = {}
    rtn['dynamite']['commit'] = bbuild.get_build_commit()
    rtn['dynamite']['branch'] = bbuild.get_build_branch()
    rtn['dynamite']['version'] = bbuild.get_build_version()
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

    dnm_version = info['dynamite']['version']
    dnm_commit = info['dynamite']['commit']
    dnm_branch = info['dynamite']['branch']
    PETSc_v = '.'.join(
        [str(info['PETSc'][k]) for k in ['major', 'minor', 'subminor']]
    )
    SLEPc_v = '.'.join(
        [str(info['SLEPc'][k]) for k in ['major', 'minor', 'subminor']]
    )

    rtn = f'dynamite version {dnm_version} (commit {dnm_commit} on branch ' +\
        f'"{dnm_branch}") built with PETSc {PETSc_v} and SLEPc {SLEPc_v}'
    return rtn


def track_memory():
    '''
    Begin tracking memory usage for a later call to ``get_memory_usage(..., max_usage=True)``.
    '''
    from . import config
    config._initialize()
    from ._backend import bpetsc
    return bpetsc.track_memory()


def get_memory_usage(group_by='all', max_usage=False):
    '''
    Get the memory usage, in gigabytes.

    .. note::
        :meth:`track_memory` must be called before this function is called
        with ``max_usage=True``.

    .. note::
        Grouping by node only works if MPI is configured to allow shared memory between ranks on
        the same node. If it is not, it may consider each rank its own "node." Whether this is the
        case can be seen by observing whether the value returned by this function is identical for
        all ranks on the same node, or if it is instead the same as the value returned for
        ``group_by='rank'``.

    Parameters
    ----------
    group_by : str
        What ranks to sum memory usage over. Options are "rank", which will return each rank's
        individual memory usage (which may be different across ranks); "node", which will sum
        over ranks sharing the same memory (and thus again the result may differ between
        ranks); and "all", which returns the total memory usage of all ranks.

    max_usage : bool
        Instead of current memory usage, report maximum since the call to :meth:`track_memory()`.
        Note that maximum is only updated when PETSc objects are destroyed, which may be delayed
        due to garbage collection.

    Returns
    -------
    float
        The memory usage in gigabytes
    '''
    from . import config
    config._initialize()
    from ._backend import bpetsc

    if max_usage:
        local_usage = bpetsc.get_max_memory_usage()/1E9
    else:
        local_usage = bpetsc.get_cur_memory_usage()/1E9

    comm = MPI_COMM_WORLD()
    if group_by == 'rank' or comm.size == 1:
        return local_usage

    import mpi4py
    comm = comm.tompi4py()

    if group_by == 'node':
        split_comm = comm.Split_type(mpi4py.MPI.COMM_TYPE_SHARED)
    elif group_by == 'all':
        split_comm = comm
    else:
        raise ValueError(f"group_by must be 'rank', 'node', or 'all'; got '{group_by}'")

    return split_comm.allreduce(local_usage)


def get_max_memory_usage(which='all'):
    '''
    [deprecated]
    '''
    if which != 'all':
        raise ValueError('values of "which" other than "all" no longer supported')

    warnings.warn(
        "get_max_memory_usage() is deprecated; use get_memory_usage(max_usage=True) instead",
        DeprecationWarning,
        stacklevel=2
    )

    return get_memory_usage(group_by='rank', max_usage=True)


def get_cur_memory_usage(which='all'):
    '''
    [deprecated]
    '''
    if which != 'all':
        raise ValueError('values of "which" other than "all" no longer supported')

    warnings.warn(
        "get_cur_memory_usage() is deprecated; use get_memory_usage() instead",
        DeprecationWarning,
        stacklevel=2
    )

    return get_memory_usage(group_by='rank')


def complex_enabled():
    from ._backend import bbuild
    return bbuild.complex_enabled()
