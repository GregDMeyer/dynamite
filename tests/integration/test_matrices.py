'''
Integration tests ensuring that matrices are built correctly.
'''

import numpy as np

import unittest as ut
import dynamite_test_runner as dtr

from dynamite import config
from dynamite.states import State
from dynamite.tools import complex_enabled
from dynamite.operators import identity, sigmax, sigmay, index_sum
from dynamite.msc_tools import msc_dtype
from dynamite.subspaces import Auto, SpinConserve
import hamiltonians


def petsc_mat_to_np(mat):
    config._initialize()
    from petsc4py import PETSc
    PROC_0 = PETSc.COMM_WORLD.rank == 0
    dims = mat.getSize()

    if PROC_0:
        rtn = np.ndarray(dims, dtype=np.complex128)
    else:
        rtn = None

    for (i, col) in enumerate(petsc_mat_columns(mat)):
        if rtn is not None:
            rtn[:,i] = col

    return rtn


def petsc_mat_columns(mat):

    config._initialize()
    from petsc4py import PETSc
    PROC_0 = PETSc.COMM_WORLD.rank == 0

    dims = mat.getSize()

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
            yield r
        else:
            yield None


from dynamite.operators import sigmax, sigmay, sigmaz, sigma_plus, sigma_minus

class Fundamental(dtr.DynamiteTestCase):

    def test_sigmax(self):
        o = sigmax()
        o.L = 1
        o_np = petsc_mat_to_np(o.get_mat())
        correct = np.array([[0, 1],
                            [1, 0]])

        if o_np is not None:
            self.assertTrue(np.array_equal(o_np, correct), msg=str(o_np))

    def test_sigmay(self):
        o = sigmay()
        o.L = 1

        if not complex_enabled():
            with self.assertRaises(ValueError):
                petsc_mat_to_np(o.get_mat())
            return

        o_np = petsc_mat_to_np(o.get_mat())

        correct = np.array([[0, -1j],
                            [1j, 0]])
        if o_np is not None:
            self.assertTrue(np.array_equal(o_np, correct))

    def test_sigmaz(self):
        o = sigmaz()
        o.L = 1
        o_np = petsc_mat_to_np(o.get_mat())
        correct = np.array([[1, 0],
                            [0,-1]])
        if o_np is not None:
            self.assertTrue(np.array_equal(o_np, correct))

    def test_nonhermitian_error(self):
        o = sigma_plus()
        with self.assertRaises(ValueError):
            o.get_mat()

        o = sigma_minus()
        with self.assertRaises(ValueError):
            o.get_mat()


def generate_hamiltonian_tests(cls):
    for H_name, real in hamiltonians.names:
        if not complex_enabled() and not real:
            continue
        setattr(cls, 'test_'+H_name, lambda self, n=H_name: self.check_hamiltonian(n))
    return cls

@generate_hamiltonian_tests
class Compare(dtr.DynamiteTestCase):
    '''
    compare numpy matrix building to the PETSc one
    '''

    def compare_matrices(self, operator):
        np_mat = operator.to_numpy()
        for (i, col) in enumerate(petsc_mat_columns(operator.get_mat())):
            if col is not None:
                np_col = np_mat[:, i].T
                self.assertTrue(np.linalg.norm(np_col-col) < 1E-13,
                                msg=f'difference found in column {i}:\ndiff: {np_col - col}')

    def check_hamiltonian(self, H_name):
        H = getattr(hamiltonians, H_name)()
        self.compare_matrices(H)

    def test_heisenberg_spinconserve(self):
        if config.L % 2 != 0:
            self.skipTest("only for even L")

        op = index_sum(sigmax(0)*sigmax(1) + sigmay(0)*sigmay(1))

        for spinflip in ['+', '-', None]:
            with self.subTest(spinflip=spinflip):
                op.subspace = SpinConserve(config.L, config.L//2, spinflip=spinflip)
                self.compare_matrices(op)


@ut.skipIf(msc_dtype['masks'] != np.int64,
           reason='only for builds with 64 bit integers')
class LargeInt64(dtr.DynamiteTestCase):
    '''
    Tests for building matrices with 64 bit integers.
    '''
    def setUp(self):
        self.old_L = config.L
        config.L = 33
        self.H = hamiltonians.localized(33)
        self.space = Auto(self.H, 'U'+'D'*32, size_guess=33)

    def tearDown(self):
        config.L = self.old_L

    def test_identity(self):
        o = identity()
        o.L = 33
        o.subspace = self.space
        o_np = petsc_mat_to_np(o.get_mat())
        if o_np is not None:
            self.assertTrue(np.array_equal(o_np, o.to_numpy(sparse=False)),
                            msg=str(o_np))

    def test_localized(self):
        config._initialize()
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

        if H_np is not None:
            self.assertTrue(result,
                            msg=msg)

if __name__ == '__main__':
    dtr.main()
