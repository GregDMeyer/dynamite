
#$ python setup.py build_ext --inplace

import os
from os.path import join
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy
import petsc4py
import slepc4py

def configure():
    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES    = []

    PETSC_DIR  = os.environ['PETSC_DIR']
    PETSC_ARCH = os.environ['PETSC_ARCH']
    SLEPC_DIR  = os.environ['SLEPC_DIR']

    INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                     join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]

    INCLUDE_DIRS += [join(SLEPC_DIR, PETSC_ARCH, 'include'),
                     join(SLEPC_DIR, 'include')]
    LIBRARY_DIRS += [join(SLEPC_DIR, PETSC_ARCH, 'lib')]

    LIBRARIES += ['petsc','slepc']

    # PETSc/SLEPc for Python
    INCLUDE_DIRS += [petsc4py.get_include(),slepc4py.get_include()]

    # NumPy
    INCLUDE_DIRS += [numpy.get_include()]

    # backend_impl
    INCLUDE_DIRS += ['.']

    return dict(
        include_dirs=INCLUDE_DIRS,
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
        extra_objects=['backend_impl.o']
    )

extensions = [
    Extension('backend',
              sources = ['backend.pyx'],
              depends = ['backend_impl.h'],
              **configure()),
]

setup(name = "backend",
      ext_modules = cythonize(
          extensions, include_path=[petsc4py.get_include(),slepc4py.get_include()]),
     )
