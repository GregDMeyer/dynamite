#!/usr/bin/env python3

'''
This file is intended to be used to configure PETSc for dynamite.
It should be run in the PETSc root source directory.

To see all possible options, run "./configure --help" in the PETSc root directory.
You may want to pipe to "less"; it is a big help page ;)
'''

from os import environ

configure_option_dict = {

    # use native complex numbers for scalars. currently required for dynamite.
    '--with-scalar-type': 'complex',

    # GPU support
    '--with-cuda': None,  # none just means no value for this arg

    # ensure correct c++ dialect
    '--with-cxx-dialect': 'cxx14',
    '--with-cuda-dialect': 'cxx14',

    # some PETSc optimization flags
    '--with-debugging': '0',

    # compiler optimization flags
    '--COPTFLAGS': '-O3',
    '--CXXOPTFLAGS': '-O3',
    '--FOPTFLAGS': '-O3',
    '--CUDAOPTFLAGS': '-O3',

    # may need/want to adjust to match your hardware's compute capability,
    # e.g. '80' for compute capability 8.0
    # can also adjust with DNM_CUDA_ARCH environment variable (see below)
    '--with-cuda-arch': '75',
}

if 'DNM_CUDA_ARCH' in environ:
    configure_option_dict['--with-cuda-arch'] = environ['DNM_CUDA_ARCH']

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath('config'))
    import configure

    configure_options = []
    for key, val in configure_option_dict.items():
        if val is None:
            configure_options.append(key)
        else:
            configure_options.append(key+'='+val)

    configure_options += sys.argv[1:]
    configure.petsc_configure(configure_options)
