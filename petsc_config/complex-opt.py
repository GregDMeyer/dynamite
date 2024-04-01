#!/usr/bin/env python3

'''
This file is intended to be used to configure PETSc for dynamite.
It should be run in the PETSc root source directory.

To see all possible options, run "./configure --help" in the PETSc root directory.
You may want to pipe to "less"; it is a big help page ;)
'''

configure_options = [

    # don't try to configure CUDA here
    '--with-cuda=0',
    '--with-cudac=0',

    # some optimization flags
    '--with-debugging=0',

    # compiler optimization flags
    '--COPTFLAGS=-O3',
    '--CXXOPTFLAGS=-O3',
    '--FOPTFLAGS=-O3',

    # use native complex numbers for scalars. currently required for dynamite.
    '--with-scalar-type=complex',

    # download extra packages for shift-invert eigensolving (solving for the
    # middle of the spectrum) with MPI.
    # not required if you won't use that feature
    #'--with-scalapack', # if you already have scalapack installed
    '--download-scalapack',
    '--download-mumps',

    # required to work with spin chains larger than 31 spins
    #'--with-64-bit-indices',

    # uncomment if you don't have an MPI implementation already installed
    #'--download-mpich',

]

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath('config'))
    import configure

    configure_options += sys.argv[1:]

    if '--download-mumps' not in configure_options:
        print('Building without MUMPS.\nIf you want to solve for interior eigenvalues, '
              'enable MUMPS and Scalapack by editing this script.')

    configure.petsc_configure(configure_options)
