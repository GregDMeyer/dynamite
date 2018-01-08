
from unittest.mock import Mock
from sys import modules
from dynamite import config

config.mock_backend = True

modules['petsc4py.PETSc'] = Mock()
modules['slepc4py.SLEPc'] = Mock()

from petsc4py import PETSc

PETSc.COMM_WORLD.rank = 0
