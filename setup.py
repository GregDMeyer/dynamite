
from sys import version
if version[0] != '3':
    raise RuntimeError('Dynamite is written for Python 3. Please install'
                       'for that version of Python.')
# TODO: do we need a particular sub-version?

from setuptools import setup

# TODO: package the C files along with the pyx for non-development users?
from Cython.Build import cythonize

import petsc4py
import slepc4py

from config_extensions import extensions, MakeBuildExt, write_build_headers

write_build_headers()

setup(
    name            = "dynamite",
    version         = "0.0.3",
    author          = "Greg Meyer",
    author_email    = "gregory.meyer@berkeley.edu",
    description     = "Fast numerics for large quantum spin chains.",
    packages        = ['dynamite'],
    classifiers = [
        "Development Status :: 4 - Beta",
    ],
    ext_modules = cythonize(
        extensions(), include_path=[petsc4py.get_include(), slepc4py.get_include()]
        ),
    cmdclass = {'build_ext' : MakeBuildExt}
)
