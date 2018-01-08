
"""
The purpose of this module is to allow lazy loading of packages, such that
we do not initialize PETSc/MPI unnecessarily.
"""

from importlib import import_module
from . import config

def get_import(name):

    need_init = ['petsc4py',
                 'slepc4py',
                 'backend']

    package = None

    if any(name.startswith(m) for m in need_init) and not config.mock_backend:
        config.initialize()

    if name == 'backend':
        name = '..backend.backend'
        package = __name__
        # TODO: figure out how to best import it down the source tree, and build
        # two separate backends for mpi and serial

    return import_module(name,package=package)
