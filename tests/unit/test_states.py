
import unittest as ut
from unittest.mock import Mock, MagicMock, ANY
import numpy as np

from dynamite import config
config.initialize()

from petsc4py import PETSc

# mock a vector
PETSc.Vec = MagicMock()

from dynamite.states import State

class StrToIdx(ut.TestCase):

    test_cases = [
        ('UUUDD', 5, 107),
        ('DDUDD', 5, 104),
        ('UDDDUDUD', 8, 181),
        (49, 6, 149),
    ]

    fail_cases = [
        ('UUFDDD', 6, lambda x: x),     # bad character
        ('UDDUD',  6, lambda x: x),     # wrong size
        ('UDDUD',  6, lambda x: np.nan) # not in subspace
    ]

    def test_good(self):
        state_to_idx = lambda x: x+100
        for s, L, c in self.test_cases:
            with self.subTest(s = s):
                self.assertEqual(State._str_to_idx(s, state_to_idx, L), c)

    def test_fail(self):
        for s, L, s2i in self.fail_cases:
            with self.assertRaises(ValueError):
                State._str_to_idx(s, L, s2i)

class SetValues(ut.TestCase):

    def setUp(self):
        self.s = State(L = 5)

        self.s._subspace = Mock()
        self.s._subspace.state_to_idx = lambda x: x+1

        self.s.vec.getOwnershipRange = Mock(return_value=(10,20))

    def test_product_string(self):
        '''
        setting product states via string
        '''

        subtests = [
            ('DDDDD', (1,1)),
            ('DUUDD', (7,1)),
            ('UUUUU', (32,1)),
            ('DUUUD', (15,1)),
        ]

        istart, iend = self.s.vec.getOwnershipRange()
        for i,a in subtests:
            with self.subTest(i=i,a=a):
                self.s.set_product(i)

                if istart <= a[0] < iend:
                    self.s.vec.__setitem__.assert_called_with(*a)

        self.s.vec.set.assert_called_with(0)

    def test_product_int(self):
        '''
        setting product states via int
        '''

        subtests = [
            (0,  (1,1)),
            (6,  (7,1)),
            (19, (20,1)),
            (31, (32,1)),
            (10, (11,1)),
        ]

        istart, iend = self.s.vec.getOwnershipRange()
        for i,a in subtests:
            with self.subTest(i=i,a=a):
                self.s.set_product(i)

                if istart <= a[0] < iend:
                    self.s.vec.__setitem__.assert_called_with(*a)

        self.s.vec.set.assert_called_with(0)

    def test_random(self):
        '''
        setting random state
        '''

        subtests = [None,0,5]

        for seed in subtests:
            with self.subTest(seed=seed):
                self.s.set_random(seed=seed)
                self.s.vec.__setitem__.assert_called_with(slice(10,20,None),ANY)

if __name__ == '__main__':
    ut.main()
