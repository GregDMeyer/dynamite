'''
Integration tests ensuring that matrices are built correctly.
'''

import numpy as np

import unittest as ut
import dynamite_test_runner as dtr

from dynamite import config
from dynamite.states import State

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
        r = State._to_numpy(column)

        if PROC_0:
            rtn[:,i] = r

    if PROC_0:
        return rtn
    else:
        return None

from dynamite.operators import sigmax, sigmay, sigmaz

class Fundamental(dtr.DynamiteTestCase):

    def proc0_assert_true(self, *args, **kwargs):
        config.initialize()
        from petsc4py import PETSc
        if PETSc.COMM_WORLD.rank == 0:
            self.assertTrue(*args, **kwargs)

    def test_sigmax(self):
        o = sigmax()
        o.L = 1
        o_np = petsc_mat_to_np(o.get_mat())
        correct = np.array([[0, 1],
                            [1, 0]])
        self.proc0_assert_true(np.array_equal(o_np, correct), msg=str(o_np))

    def test_sigmay(self):
        o = sigmay()
        o.L = 1
        o_np = petsc_mat_to_np(o.get_mat())
        correct = np.array([[0, -1j],
                            [1j, 0]])
        self.proc0_assert_true(np.array_equal(o_np, correct))

    def test_sigmaz(self):
        o = sigmaz()
        o.L = 1
        o_np = petsc_mat_to_np(o.get_mat())
        correct = np.array([[1, 0],
                            [0,-1]])
        self.proc0_assert_true(np.array_equal(o_np, correct))

from dynamite.operators import identity
from dynamite.msc_tools import msc_dtype
from dynamite.subspaces import Auto
from hamiltonians import localized
@ut.skipIf(msc_dtype['masks'] != np.int64,
           reason='only for builds with 64 bit integers')
class LargeInt64(dtr.DynamiteTestCase):
    '''
    Tests for building matrices with 64 bit integers.
    '''

    def proc0_assert_true(self, *args, **kwargs):
        # TODO: help this not hang if we fail a test by bcast
        config.initialize()
        from petsc4py import PETSc
        if PETSc.COMM_WORLD.rank == 0:
            self.assertTrue(*args, **kwargs)

    def setUp(self):
        self.H = localized(33)
        self.space = Auto(self.H, 'U'+'D'*32, size_guess=33)

    def test_identity(self):
        o = identity()
        o.L = 33
        o.subspace = self.space
        o_np = petsc_mat_to_np(o.get_mat())
        self.proc0_assert_true(np.array_equal(o_np, o.to_numpy(sparse=False)),
                               msg=str(o_np))

    def test_localized(self):
        config.initialize()
        from petsc4py import PETSc

        self.H.subspace = self.space
        H_dnm = petsc_mat_to_np(self.H.get_mat())
        H_np = self.H.to_numpy(sparse=False)

        result = np.array_equal(H_dnm, H_np)
        msg = ''
        if not result:
            if PETSc.COMM_WORLD.rank == 0:
                diffs = np.where(np.abs(H_np-H_dnm) > 1E-15)
                msg += '\ndiff:\n'
                for row_idx, col_idx in zip(diffs[0][:30], diffs[1][:30]):
                    msg += '(%d, %d): np: %s dnm: %s\n' % (
                        row_idx, col_idx,
                        str(H_np[row_idx, col_idx]),
                        str(H_dnm[row_idx, col_idx])
                    )
            else:
                msg = ''

        self.proc0_assert_true(result,
                               msg=msg)

if __name__ == '__main__':
    dtr.main()
