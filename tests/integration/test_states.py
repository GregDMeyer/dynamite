'''
Integration tests for states.
'''

import unittest as ut
import numpy as np

import dynamite_test_runner as dtr

from dynamite.states import State

class RandomSeed(dtr.DynamiteTestCase):

    def test_generation(self):
        '''
        Make sure that different processors get the same random seed.
        '''

        from dynamite import config
        config._initialize()
        from petsc4py import PETSc
        comm = PETSc.COMM_WORLD.tompi4py()
        seed = State.generate_time_seed()

        all_seeds = comm.gather(seed, root = 0)

        if comm.rank == 0:
            self.assertTrue(all(s == seed for s in all_seeds))

class ToNumpy(dtr.DynamiteTestCase):

    def setUp(self):
        from petsc4py import PETSc

        self.v = PETSc.Vec().create()
        self.v.setSizes(PETSc.COMM_WORLD.size)
        self.v.setFromOptions()
        self.v.set(-1)

        self.v[PETSc.COMM_WORLD.rank] = PETSc.COMM_WORLD.rank
        self.v.assemblyBegin()
        self.v.assemblyEnd()

    def test_to_zero(self):
        from petsc4py import PETSc
        npvec = State._to_numpy(self.v)
        if PETSc.COMM_WORLD.rank == 0:
            for i in range(PETSc.COMM_WORLD.rank):
                self.assertTrue(npvec[i] == i)
        else:
            self.assertIs(npvec, None)

    def test_to_all(self):
        from petsc4py import PETSc
        npvec = State._to_numpy(self.v, to_all = True)
        for i in range(PETSc.COMM_WORLD.rank):
            self.assertTrue(npvec[i] == i)

class PetscMethods(dtr.DynamiteTestCase):
    '''
    Tests that the methods directly included from PETSc function as intended.
    '''
    def test_norm(self):
        state = State()
        start, end = state.vec.getOwnershipRange()
        state.vec[start:end] = np.array([1]*(end-start))
        state.vec.assemblyBegin()
        state.vec.assemblyEnd()
        self.assertAlmostEqual(state.norm()**2, state.subspace.get_dimension())

    def test_normalize(self):
        state = State()
        start, end = state.vec.getOwnershipRange()
        state.vec[start:end] = np.array([1]*(end-start))
        state.vec.assemblyBegin()
        state.vec.assemblyEnd()
        state.normalize()
        self.assertTrue(state.norm() == 1)

    def test_copy_preallocate(self):
        state1 = State()
        state2 = State()
        start, end = state1.vec.getOwnershipRange()
        state1.vec[start:end] = np.arange(start, end)
        state1.vec.assemblyBegin()
        state1.vec.assemblyEnd()

        result = np.ndarray((end-start,), dtype=np.complex128)
        state1.copy(state2)
        result[:] = state2.vec[start:end]

        self.assertTrue(np.array_equal(result, np.arange(start, end)))

    def test_copy_exception_L(self):
        state1 = State()
        state2 = State(L=state1.subspace.L+1)

        with self.assertRaises(ValueError):
            state1.copy(state2)

    def test_copy_nopreallocate(self):
        state1 = State()
        start, end = state1.vec.getOwnershipRange()
        state1.vec[start:end] = np.arange(start, end)
        state1.vec.assemblyBegin()
        state1.vec.assemblyEnd()

        result = np.ndarray((end-start,), dtype=np.complex128)
        state2 = state1.copy()
        result[:] = state2.vec[start:end]

        self.assertTrue(np.array_equal(result, np.arange(start, end)))

    def test_scale(self):
        vals = [2, 3.14]
        for val in vals:
            with self.subTest(val=val):
                state = State(state='random')
                start, end = state.vec.getOwnershipRange()
                pre_values = np.ndarray((end-start,), dtype=np.complex128)
                pre_values[:] = state.vec[start:end]

                state *= val

                for i in range(start, end):
                    self.assertEqual(state.vec[i], val*pre_values[i-start])

    def test_scale_divide(self):
        val = 3.14
        state = State(state='random')
        start, end = state.vec.getOwnershipRange()
        pre_values = np.ndarray((end-start,), dtype=np.complex128)
        pre_values[:] = state.vec[start:end]

        state /= val

        for i in range(start, end):
            self.assertEqual(state.vec[i], (1/val)*pre_values[i-start])

    def test_scale_exception_ary(self):
        val = np.array([3.1, 4])
        state = State()
        with self.assertRaises(TypeError):
            state *= val

    def test_scale_exception_vec(self):
        state1 = State()
        state2 = State()
        with self.assertRaises(TypeError):
            state1 *= state2

# TODO: check state setting. e.g. setting an invalid state should fail (doesn't for Full subspace)

if __name__ == '__main__':
    dtr.main()
