#!/usr/bin/env python2

'''
This file is intended to be used to configure PETSc for dynamite.
It should be run in the PETSc root source directory.

To see all possible options, run "./configure --help" in the PETSc root directory.
You may want to pipe to "less"; it is a big help page ;)
'''

configure_options = [

  # a name for this PETSc build. Feel free to name it whatever you want, so that
  # you can keep track of multiple builds.
  '--with-petsc-arch=complex-opt',

  # some optimization flags
  '--with-debugging=0',
  '--with-fortran-kernels=1',

  # compiler optimization flags
  '--COPTFLAGS=-O3',
  '--CXXOPTFLAGS=-O3',
  '--FOPTFLAGS=-O3',

  # use native complex numbers for scalars. currently required for dynamite.
  '--with-scalar-type=complex',

  # download extra packages for shift-invert eigensolving (solving for the middle
  # of the spectrum). not required if you won't use that feature
  #'--with-scalapack', # if you already have scalapack installed
  '--download-scalapack',
  '--download-mumps',

  # required to work with spin chains larger than 31 spins
  #'--use-64-bit-indices',

  # uncomment if you don't have an MPI implementation already installed
  #'--download-mpich',

  # GPU support
  # dynamite's GPU support is experimental! see
  # http://www.mcs.anl.gov/petsc/features/gpus.html
  # for more information about using PETSc with GPUs.
  #'--with-cuda',

  ]

if __name__ == '__main__':
  import sys, os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
