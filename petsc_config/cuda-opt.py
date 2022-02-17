#!/usr/bin/env python3

'''
This file is intended to be used to configure PETSc for dynamite.
It should be run in the PETSc root source directory.

To see all possible options, run "./configure --help" in the PETSc root directory.
You may want to pipe to "less"; it is a big help page ;)
'''

configure_options = [

    # some optimization flags
    '--with-debugging=0',
    '--with-fortran-kernels=1',

    # compiler optimization flags
    '--COPTFLAGS=-O3',
    '--CXXOPTFLAGS=-O3',
    '--FOPTFLAGS=-O3',
    '--CUDAOPTFLAGS=-O3',

    # use native complex numbers for scalars. currently required for dynamite.
    '--with-scalar-type=complex',

    # GPU support
    # dynamite's GPU support is experimental! see
    # http://www.mcs.anl.gov/petsc/features/gpus.html
    # for more information about using PETSc with GPUs.
    '--with-cuda',

    # may need/want to adjust to match your hardware's compute capability,
    # e.g. '80' for compute capability 8.0
    '--with-cuda-arch=75',

    # ensure correct c++ dialect
    '--with-cxx-dialect=cxx14',
    '--with-cuda-dialect=cxx14'
]

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath('config'))
    import configure

    configure_options += sys.argv[1:]
    configure.petsc_configure(configure_options)
