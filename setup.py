
import os
from os.path import join
from sys import version
from subprocess import check_output, CalledProcessError
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

import numpy
import petsc4py
import slepc4py

if version[0] != '3':
    raise RuntimeError('Dynamite is written for Python 3. Please install'
                       'for that version of Python.')

def configure():

    if any(e not in os.environ for e in ['PETSC_DIR','PETSC_ARCH','SLEPC_DIR']):
        raise ValueError('Must set environment variables PETSC_DIR, '
                         'PETSC_ARCH and SLEPC_DIR before installing! '
                         'If executing with sudo, you may want the -E '
                         'flag to pass environment variables through '
                         'sudo.')

    PETSC_DIR  = os.environ['PETSC_DIR']
    PETSC_ARCH = os.environ['PETSC_ARCH']
    SLEPC_DIR  = os.environ['SLEPC_DIR']

    include = []
    lib_paths = []

    include += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                join(PETSC_DIR, 'include')]
    lib_paths += [join(PETSC_DIR, PETSC_ARCH, 'lib')]

    include += [join(SLEPC_DIR, PETSC_ARCH, 'include'),
                join(SLEPC_DIR, 'include')]
    lib_paths += [join(SLEPC_DIR, PETSC_ARCH, 'lib')]

    libs = ['petsc','slepc']

    # python package includes
    include += [petsc4py.get_include(),
                slepc4py.get_include(),
                numpy.get_include()]

    return dict(
        include_dirs=include,
        libraries=libs,
        library_dirs=lib_paths,
        runtime_library_dirs=lib_paths,
        extra_objects=['dynamite/backend/backend_impl.o',
                       'dynamite/backend/cuda_shell.o']
    )

def extensions():
    return [
        Extension('dynamite.backend.backend',
                  sources = ['dynamite/backend/backend.pyx'],
                  depends = ['dynamite/backend/backend_impl.h',
                             'dynamite/backend/cuda_shell.h'],
                  **configure()),
    ]

class MakeBuildExt(build_ext):
    def run(self):
        # build the backend_impl.o object file
        make = check_output(['make','backend_impl.o'],cwd='dynamite/backend')
        print(make.decode())

        # try to build the cuda backend, if it doesn't work (probably because
        # cuda isn't present), no worries
        try:
            make = check_output(['make','cuda_shell.o'],cwd='dynamite/backend')
            print(make.decode())
        except CalledProcessError:
            print('CUDA compilation failed. GPU shell matrices disabled.')

        build_ext.run(self)

setup(
    name = "dynamite",
    version = "0.0.2",
    author = "Greg Meyer",
    author_email = "gregory.meyer@berkeley.edu",
    description = "Fast direct evolution of quantum spin chains.",
    packages=['dynamite'],
    classifiers=[
        "Development Status :: 4 - Beta",
    ],
    ext_modules = cythonize(
        extensions(), include_path=[petsc4py.get_include(),slepc4py.get_include()]
        ),
    cmdclass = {'build_ext' : MakeBuildExt}
)
