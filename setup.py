
import os
from subprocess import check_output
from glob import glob

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

import numpy


def main():
    env_vars = ['PETSC_DIR', 'PETSC_ARCH', 'SLEPC_DIR']
    if any(e not in os.environ for e in env_vars):
        raise RuntimeError('Must set environment variables PETSC_DIR, '
                           'PETSC_ARCH and SLEPC_DIR before installing!')
    write_build_headers()

    setup(
        ext_modules=cythonize(
            extensions(), include_path=get_cython_includes()
            ),
        cmdclass={'build_ext': MakeBuildExt},
        package_dir={'': 'src'}
    )


def get_cython_includes():
    # the following two replace petsc4py.get_include() and
    # slepc4py.get_include(), so we don't need those two installed
    # during the build process
    return [
        os.path.join(
            os.environ['PETSC_DIR'],
            'src/binding/petsc4py/src/include'
        ),
        os.path.join(
            os.environ['SLEPC_DIR'],
            'src/binding/slepc4py/src/include'
        )
    ]


def write_build_headers():
    '''
    Write a Cython include file with some constants that become
    hardcoded into the backend build.
    '''
    print('Writing header files...')

    commit = check_output(['git', 'describe', '--always'],
                          cwd=os.path.dirname(os.path.realpath(__file__)),
                          universal_newlines=True).strip()
    branch = check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                          cwd=os.path.dirname(os.path.realpath(__file__)),
                          universal_newlines=True).strip()
    version = open('VERSION').read().strip()

    header_path = os.path.join(
        os.path.dirname(__file__),
        'src/dynamite/_backend/config.pxi'
    )
    with open(header_path, 'w') as f:
        f.write('DEF USE_CUDA = %d\n' % int(check_cuda()))
        f.write('DEF DNM_BRANCH = "%s"\n' % branch)
        f.write('DEF DNM_COMMIT = "%s"\n' % commit)
        f.write('DEF DNM_VERSION = "%s"\n' % version)


def extensions():

    extension_names = [
        'bsubspace',
        'bbuild',
        'bpetsc'
    ]

    header_only = {
        'bsubspace',
    }

    cython_only = {
        'bbuild',
    }

    exts = []
    for name in extension_names:

        depends = []
        object_files = []
        extra_compile_args = []
        petsc_vars = get_petsc_variables()
        CC_FLAGS = list(filter(None, petsc_vars["CC_FLAGS"].split(' ')))
        CXX_FLAGS = list(filter(None, petsc_vars["CXX_FLAGS"].split(' ')))

        if name not in cython_only:
            depends += [f'src/dynamite/_backend/{name}_impl.h']
            if name not in header_only:
                depends += [f'src/dynamite/_backend/{name}_impl.c']
                object_files = [f'src/dynamite/_backend/{name}_impl.o']
            else:
                extra_compile_args += CXX_FLAGS

        else:
            extra_compile_args += CC_FLAGS

        if name == 'bpetsc':
            extra_compile_args += CC_FLAGS
            depends += ['src/dynamite/_backend/bsubspace.pxd'
                        'src/dynamite/_backend/bcuda_impl.h',
                        'src/dynamite/_backend/bcuda_impl.cu',
                        'src/dynamite/_backend/shellcontext.h',
                        'src/dynamite/_backend/bsubspace_impl.h']
            if check_cuda():
                object_files += ['src/dynamite/_backend/bcuda_impl.o']

        exts += [
            Extension(f'dynamite._backend.{name}',
                      sources=[f'src/dynamite/_backend/{name}.pyx'],
                      depends=depends,
                      extra_objects=object_files,
                      extra_compile_args=extra_compile_args,
                      **configure_paths())
        ]

    return exts


def configure_paths():

    PETSC_DIR = os.environ['PETSC_DIR']
    PETSC_ARCH = os.environ['PETSC_ARCH']
    SLEPC_DIR = os.environ['SLEPC_DIR']

    includes = []
    libs = []

    includes += [os.path.join(PETSC_DIR, PETSC_ARCH, 'include'),
                 os.path.join(PETSC_DIR, 'include')]
    libs += [os.path.join(PETSC_DIR, PETSC_ARCH, 'lib')]

    includes += [os.path.join(SLEPC_DIR, PETSC_ARCH, 'include'),
                 os.path.join(SLEPC_DIR, 'include')]
    libs += [os.path.join(SLEPC_DIR, PETSC_ARCH, 'lib')]

    # python package includes
    includes += [numpy.get_include()]

    return dict(
        include_dirs=includes,
        library_dirs=libs,
        runtime_library_dirs=libs,
        libraries=['petsc', 'slepc']
    )


class MakeBuildExt(build_ext):

    def run(self):

        # first remove any old object files which may
        # correspond to a different PETSC_ARCH
        for fname in glob('src/dynamite/_backend/*.o'):
            os.remove(fname)

        # build the object files
        make = check_output(['make', 'bpetsc_impl.o'],
                            cwd='src/dynamite/_backend')
        print(make.decode(), end='')

        if check_cuda():
            make = check_output(['make', 'bcuda_impl.o'],
                                cwd='src/dynamite/_backend')
            print(make.decode(), end='')

        petsc_vars = get_petsc_variables()
        os.environ['CC'] = petsc_vars['CC']
        os.environ['CXX'] = petsc_vars['CXX']

        build_ext.run(self)


USE_CUDA = None
def check_cuda():
    '''
    Whether PETSc was built with CUDA support
    '''
    global USE_CUDA
    if USE_CUDA is not None:
        return USE_CUDA

    with open(os.path.join(os.environ['PETSC_DIR'],
                           os.environ['PETSC_ARCH'],
                           'include/petscconf.h')) as f:
        for line in f:
            if 'PETSC_HAVE_CUDA 1' in line:
                USE_CUDA = True
                break
        else:
            USE_CUDA = False

    return USE_CUDA


def get_petsc_variables():
    petsc_varfname = os.path.join(
        os.environ['PETSC_DIR'],
        os.environ['PETSC_ARCH'],
        'lib/petsc/conf/petscvariables'
    )

    petsc_vars = {}
    with open(petsc_varfname) as f:
        for line in f:
            (key, val) = line.rstrip('\r\n').split(" = ")
            petsc_vars[key] = val.strip()

    return petsc_vars


if __name__ == '__main__':
    main()
