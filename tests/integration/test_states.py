'''
Integration tests for states.
'''

import unittest as ut
from dynamite.states import State

class RandomSeed(ut.TestCase):

    def test_generation(self):
        '''
        Make sure that different processors get the same random seed.
        '''

        from petsc4py import PETSc
        comm = PETSc.COMM_WORLD.tompi4py()
        seed = State.generate_time_seed()

        all_seeds = comm.gather(seed, root = 0)

        if comm.rank == 0:
            self.assertTrue(all(s == seed for s in all_seeds))

class ToNumpy(ut.TestCase):

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

# TODO: check state setting. e.g. setting an invalid state should fail (doesn't for Full subspace)

if __name__ == '__main__':
    ut.main()
