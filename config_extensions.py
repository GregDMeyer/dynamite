
from os import environ
from os.path import join, dirname, realpath
from subprocess import check_output
from shutil import which

from setuptools import Extension
from setuptools.command.build_ext import build_ext

import numpy
import petsc4py
import slepc4py

extension_names = [
    'bsubspace',
    'bbuild'
    #'bpetsc'
]

header_only = {
    'bsubspace',
}

cython_only = {
    'bbuild'
}

def extensions():

    exts = []
    for name in extension_names:

        depends = []
        object_files = []
        extra_args = {
            'include_dirs' : [numpy.get_include()] + get_petsc_includes()
        }

        if name not in cython_only:
            depends += ['dynamite/_backend/{name}_impl.h'.format(name=name)]
            if name not in header_only:
                depends += ['dynamite/_backend/{name}_impl.c'.format(name=name)]
                object_files = ['dynamite/_backend/{name}_impl.o'.format(name=name)]

        if name == 'bpetsc':
            depends += ['dynamite/_backend/bcuda_impl.h',
                        'dynamite/_backend/bcuda_impl.cu',
                        'dynamite/_backend/shellcontext.h',
                        'dynamite/_backend/bsubspace_impl.h']
            object_files += ['dynamite/_backend/cuda_impl.o'.format(name=name)]
            extra_args = configure_petsc_backend()

        exts = [
            Extension('dynamite._backend.{name}'.format(name=name),
                      sources = ['dynamite/_backend/{name}.pyx'.format(name=name)],
                      depends = depends,
                      **extra_args)
            for name in extension_names if name != 'bpetsc'
        ]

    return exts

def have_nvcc():
    '''
    Whether the nvcc compiler can be found.
    '''
    return which('nvcc') is not None

def write_build_headers():
    '''
    Write a Cython include file with some constants that become
    hardcoded into the backend build.
    '''
    print('Writing header files...')
    with open(join(dirname(__file__), 'dynamite', '_backend', 'config.pxi'), 'w') as f:

        f.write('DEF USE_CUDA = %d\n' % int(have_nvcc()))

        dnm_version = check_output(['git', 'describe', '--always'],
                                   cwd = dirname(realpath(__file__)),
                                   universal_newlines = True).strip()
        f.write('DEF DNM_VERSION = "%s"\n' % dnm_version)

        dnm_version = check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                   cwd = dirname(realpath(__file__)),
                                   universal_newlines = True).strip()
        f.write('DEF DNM_BRANCH = "%s"\n' % dnm_version)

def get_petsc_includes():
    if any(e not in environ for e in ['PETSC_DIR', 'PETSC_ARCH']):
        raise ValueError('Must set environment variables PETSC_DIR '
                         'and PETSC_ARCH before installing! '
                         'If executing with sudo, you may want the -E '
                         'flag to pass environment variables through '
                         'sudo.')

    PETSC_DIR  = environ['PETSC_DIR']
    PETSC_ARCH = environ['PETSC_ARCH']

    include = [join(PETSC_DIR, PETSC_ARCH, 'include'),
               join(PETSC_DIR, 'include')]

    return include

def configure_petsc_backend():

    if any(e not in environ for e in ['PETSC_DIR', 'PETSC_ARCH', 'SLEPC_DIR']):
        raise ValueError('Must set environment variables PETSC_DIR, '
                         'PETSC_ARCH and SLEPC_DIR before installing! '
                         'If executing with sudo, you may want the -E '
                         'flag to pass environment variables through '
                         'sudo.')

    PETSC_DIR  = environ['PETSC_DIR']
    PETSC_ARCH = environ['PETSC_ARCH']
    SLEPC_DIR  = environ['SLEPC_DIR']

    include = []
    lib_paths = []

    include += get_petsc_includes
    lib_paths += [join(PETSC_DIR, PETSC_ARCH, 'lib')]

    include += [join(SLEPC_DIR, PETSC_ARCH, 'include'),
                join(SLEPC_DIR, 'include')]
    lib_paths += [join(SLEPC_DIR, PETSC_ARCH, 'lib')]

    libs = ['petsc','slepc']

    # python package includes
    include += [petsc4py.get_include(),
                slepc4py.get_include(),
                numpy.get_include()]

    object_files = ['dynamite/backend/bpetsc_impl.o']

    # check if we should link to the CUDA code
    if have_nvcc():
        object_files = ['dynamite/backend/cuda_shell.o'] + object_files

    return dict(
        include_dirs=include,
        libraries=libs,
        library_dirs=lib_paths,
        runtime_library_dirs=lib_paths,
        extra_objects=object_files
    )

class MakeBuildExt(build_ext):

    def run(self):

        write_build_headers()

        # build the object files
        for name in extension_names:
            if name in header_only | cython_only:
                continue

            make = check_output(['make', '{name}_impl.o'.format(name=name)],
                                cwd='dynamite/_backend')
            print(make.decode())

        # if we have nvcc, build the CUDA backend
        if have_nvcc():
            make = check_output(['make', 'bcuda_impl.o'], cwd='dynamite/_backend')
            print(make.decode())

        # get the correct compiler from SLEPc
        # there is probably a more elegant way to do this
        makefile = 'include ${SLEPC_DIR}/lib/slepc/conf/slepc_common\n' + \
                   'print_compiler:\n\t$(CC)'
        CC = check_output(['make', '-n', '-f', '-', 'print_compiler'],
                          input = makefile, encoding = 'utf-8')

        # now set environment variables to that compiler
        if 'CC' in environ:
            _old_CC = environ['CC']
        else:
            _old_CC = None

        environ['CC'] = CC

        try:
            build_ext.run(self)
        finally:
            # set CC back to its old value
            if _old_CC is not None:
                environ['CC'] = _old_CC
            else:
                environ.pop('CC')
