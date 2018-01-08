
import unittest as ut
from unittest.mock import Mock, MagicMock, ANY

import mock_backend
from petsc4py import PETSc

# mock a vector
PETSc.Vec = MagicMock()

from dynamite.states import State

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
            ('DDDDD',(1,1)),
            ('DUUDD',(7,1)),
            ('UUUUU',(32,1))
        ]

        for i,a in subtests:
            with self.subTest(i=i,a=a):
                self.s.set_product(i)
                self.s.vec.__setitem__.assert_called_with(*a)

        self.s.vec.set.assert_called_with(0)

    def test_product_string_exceptions(self):
        '''
        exceptions from string product states
        '''

        subtests = [
            ('DUDUDU'),
            ('DUDU'),
            ('DUDUDF')
        ]
        for i in subtests:
            with self.subTest(i=i):
                with self.assertRaises(ValueError):
                    self.s.set_product(i)

    def test_product_int(self):
        '''
        setting product states via int
        '''

        subtests = [
            (0,(1,1)),
            (6,(7,1)),
            (31,(32,1))
        ]

        for i,a in subtests:
            with self.subTest(i=i,a=a):
                self.s.set_product(i)
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
