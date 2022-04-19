'''
Various tools useful for writing and analyzing dynamite programs.
'''

def mpi_print(*args, rank=0, **kwargs):
    '''
    Print from only a single MPI rank, default rank 0.

    Aside from the extra "rank" keywork argument, call signature is the same
    as Python 3's ``print()`` function.
    '''
    from . import config
    config._initialize()
    from petsc4py import PETSc

    if PETSc.COMM_WORLD.rank == rank:
        print(*args, **kwargs)

def get_version():
    '''
    Gets the version information for dynamite, and the PETSc and SLEPc libraries it's built on.

    Returns
    -------

    dict
        A dictionary with the keys 'PETSc', 'SLEPc', and 'dynamite', each of which contains version
        information for the respective library.
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
    rtn['dynamite']['commit'] = bbuild.get_build_version()
    rtn['dynamite']['branch'] = bbuild.get_build_branch()
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

def track_memory():
    '''
    Begin tracking memory usage for a later call to :meth:`get_max_memory_usage`.
    '''
    from . import config
    config._initialize()
    from ._backend import bpetsc
    return bpetsc.track_memory()

def get_max_memory_usage(which='all'):
    '''
    Get the maximum memory usage up to this point. Only updated whenever
    objects are destroyed (e.g. with :meth:`dynamite.operators.Operator.destroy_mat`)

    .. note::
        :meth:`track_memory` must be called before this function is called,
        and the option ``'-malloc'`` must be supplied to PETSc at runtime if
        ``which == 'petsc'``.

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
    from . import config
    config._initialize()
    from ._backend import bpetsc
    return bpetsc.get_max_memory_usage(which=which)

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
    from . import config
    config._initialize()
    from ._backend import bpetsc
    return bpetsc.get_cur_memory_usage(which=which)

def complex_enabled():
    from ._backend import bbuild
    return bbuild.complex_enabled()
