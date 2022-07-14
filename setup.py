
from sys import version
if version[0] != '3':
    raise RuntimeError('Dynamite is written for Python 3. Please install'
                       'for that version of Python.')

from setuptools import setup

from Cython.Build import cythonize

import petsc4py
import slepc4py

from config_extensions import extensions, MakeBuildExt, write_build_headers

write_build_headers()

setup(
    name            = "dynamite",
    version         = open('VERSION').read().strip(),
    author          = "Greg Kahanamoku-Meyer",
    author_email    = "gkm@berkeley.edu",
    description     = "Fast numerics for large quantum spin chains.",
    package_dir     = {"": "src"},
    packages        = ['dynamite'],
    classifiers = [
        "Development Status :: 4 - Beta",
    ],
    ext_modules = cythonize(
        extensions(), include_path=[petsc4py.get_include(), slepc4py.get_include()]
        ),
    cmdclass = {'build_ext' : MakeBuildExt}
)
