#!/usr/bin/env python2

'''
This file is intended to be used to configure PETSc for dynamite.
It should be run in the PETSc root source directory.
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
  '--CUDAFLAGS=-arch=sm_35',

  # use native complex numbers for scalars. currently required for dynamite.
  '--with-scalar-type=complex',

  # GPU support
  # dynamite's GPU support is experimental! see
  # http://www.mcs.anl.gov/petsc/features/gpus.html
  # for more information about using PETSc with GPUs.
  '--with-cuda',

  ]

if __name__ == '__main__':
  import sys, os
  sys.path.insert(0, os.path.abspath('config'))

  configure_options += sys.argv[1:]

  import configure
  configure.petsc_configure(configure_options)
