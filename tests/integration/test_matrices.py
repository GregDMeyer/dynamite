'''
Integration tests ensuring that matrices are built correctly.
'''

import unittest as ut
import numpy as np

from dynamite import config
from dynamite.tools import vectonumpy

def petsc_mat_to_np(mat):

    config.initialize()
    from petsc4py import PETSc
    PROC_0 = PETSc.COMM_WORLD.rank == 0

    dims = mat.getSize()

    if PROC_0:
        rtn = np.ndarray(dims,dtype=np.complex128)

    select = PETSc.Vec().create()
    select.setSizes(dims[1])
    select.setFromOptions()

    column = PETSc.Vec().create()
    column.setSizes(dims[0])
    column.setFromOptions()

    select.set(0)
    for i in range(dims[0]):
        if i > 0:
            select.setValue(i-1,0)
        select.setValue(i,1)
        select.assemblyBegin()
        select.assemblyEnd()
        mat.mult(select, column)
        r = vectonumpy(column)

        if PROC_0:
            rtn[:,i] = r

    if PROC_0:
        return rtn
    else:
        return None

from dynamite.operators import sigmax, sigmay, sigmaz

class Fundamental(ut.TestCase):

    def proc0_assert_true(self, *args, **kwargs):
        # TODO: help this not hang if we fail a test by bcast
        config.initialize()
        from petsc4py import PETSc
        if PETSc.COMM_WORLD.rank == 0:
            self.assertTrue(*args, **kwargs)

    def test_sigmax(self):
        o = sigmax()
        o_np = petsc_mat_to_np(o.get_mat())
        correct = np.array([[0, 1],
                            [1, 0]])
        self.proc0_assert_true(np.array_equal(o_np, correct))

    def test_sigmay(self):
        o = sigmay()
        o_np = petsc_mat_to_np(o.get_mat())
        correct = np.array([[0, -1j],
                            [1j, 0]])
        self.proc0_assert_true(np.array_equal(o_np, correct))

    def test_sigmaz(self):
        o = sigmaz()
        o_np = petsc_mat_to_np(o.get_mat())
        correct = np.array([[1, 0],
                            [0,-1]])
        self.proc0_assert_true(np.array_equal(o_np, correct))

if __name__ == '__main__':
    ut.main()
