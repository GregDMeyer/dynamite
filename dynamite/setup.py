#!/usr/bin/env python

#$ python setup.py build_ext --inplace

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

    # PETSc
    import os
    PETSC_DIR  = os.environ['PETSC_DIR']
    PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    SLEPC_DIR  = os.environ['SLEPC_DIR']
    from os.path import join, isdir
    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                         join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
    else:
        if PETSC_ARCH: pass # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]
    if SLEPC_DIR and isdir(join(SLEPC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(SLEPC_DIR, PETSC_ARCH, 'include'),
                         join(SLEPC_DIR, 'include')]
        LIBRARY_DIRS += [join(SLEPC_DIR, PETSC_ARCH, 'lib')]
    LIBRARIES += ['petsc','slepc']

    # PETSc/SLEPc for Python
    INCLUDE_DIRS += [petsc4py.get_include(),slepc4py.get_include()]

    # NumPy
    INCLUDE_DIRS += [numpy.get_include()]

    return dict(
        include_dirs=INCLUDE_DIRS + [os.curdir],
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
    )

extensions = [
    Extension('backend',
              sources = ['backend.pyx',
                         'backend_impl.c'],
              depends = ['backend_impl.h'],
              **configure()),
]

setup(name = "backend",
      ext_modules = cythonize(
          extensions, include_path=[petsc4py.get_include(),slepc4py.get_include()]),
     )
