
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
        ('DDDUU', 5, 7),
        ('UUDUU', 5, 4),
        ('DUUUDUDU', 8, 81),
        ('11100', 5, 7),
        ('00100', 5, 4),
        ('10001010', 8, 81),
        (49, 6, 49),
    ]

    fail_cases = [
        ('DDFUUU', 6),     # bad character
        ('DUUDU',  6),     # wrong size
        (85,       6),     # wrong size
    ]

    def test_good(self):
        for s, L, c in self.test_cases:
            with self.subTest(s = s):
                self.assertEqual(State.str_to_state(s, L), c)

    def test_fail(self):
        for s, L in self.fail_cases:
            with self.assertRaises(ValueError):
                State.str_to_state(s, L)


class SetL(ut.TestCase):

    def test_L_direct(self):
        s = State(L=5)
        self.assertEqual(s.L, 5)

    def test_L_subspace(self):
        from dynamite.subspaces import Subspace
        subspace = Mock(spec=Subspace)
        subspace.L = 5

        s = State(subspace=subspace)
        self.assertEqual(s.L, 5)


class SetValues(ut.TestCase):

    def setUp(self):
        self.s = State(L = 5)

        self.s._subspace = Mock()
        self.s._subspace.state_to_idx = lambda x: x+1
        self.s._subspace.L = 5

        self.s.vec.getOwnershipRange = Mock(return_value=(10,20))

    def test_product_string(self):
        '''
        setting product states via string
        '''

        subtests = [
            ('UUUUU', (1,1)),
            ('UDDUU', (7,1)),
            ('DDDDD', (32,1)),
            ('UDDDU', (15,1)),
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


class PrettyPrint(ut.TestCase):

    def test_product_state(self):
        subtests = [
            ('0000', True),
            ('1111', True),
            ('1000', True),
            ('0110', True),
            ('UUUU', False),
            ('DDDD', False),
            ('DUUU', False),
            ('UDDU', False),
        ]

        for s, binary in subtests:
            with self.subTest(s=s):
                state = State(L=4)
                state._vec = np.zeros(16, dtype=np.complex128)
                idx = int(s.replace('D', '1').replace('U', '0')[::-1], base=2)
                state._vec[idx] = 1
                state.repr_binary = binary
                state.set_initialized()

                self.assertEqual(
                    str(state),
                    '|' + s + '>'
                )

                s = s.replace('U', r'\uparrow')
                s = s.replace('D', r'\downarrow')
                self.assertEqual(
                    state._repr_latex_(),
                    r'$\left|' + s + r'\right>$'
                )

    def test_product_state_mixed(self):
        subtests = [
            'UU00',
            'D11D',
            '1UUU',
            '0DDU'
        ]

        for s in subtests:
            with self.subTest(s=s):
                state = State(L=4)
                state._vec = np.zeros(16, dtype=np.complex128)
                idx = int(s.replace('D', '1').replace('U', '0')[::-1], base=2)
                state._vec[idx] = 1
                state.set_initialized()

                s = s.replace('D', '1').replace('U', '0')
                self.assertEqual(
                    str(state),
                    '|' + s + '>'
                )
                self.assertEqual(
                    state._repr_latex_(),
                    r'$\left|' + s + r'\right>$'
                )

    def test_superposition_real(self):
        state = State(L=4)
        state._vec = np.zeros(16, dtype=np.complex128)
        state._vec[1] = 1
        state._vec[2] = 0.5
        state.set_initialized()

        self.assertEqual(
            str(state),
            '1.000|1000> + 0.500|0100>'
        )
        self.assertEqual(
            state._repr_latex_(),
            r'$1.000\left|1000\right> + 0.500\left|0100\right>$'
        )

    def test_superposition_complex(self):
        state = State(L=4)
        state._vec = np.zeros(16, dtype=np.complex128)
        state._vec[1] = 1j
        state._vec[2] = 0.5
        state.set_initialized()

        self.assertEqual(
            str(state),
            '(0.000+1.000j)|1000> + (0.500+0.000j)|0100>'
        )
        self.assertEqual(
            state._repr_latex_(),
            r'$(0.000+1.000j)\left|1000\right> + '
            r'(0.500+0.000j)\left|0100\right>$'
        )

    def test_subspace(self):
        state = State(L=4)
        state._subspace = Mock()
        state._subspace.state_to_idx = lambda x: x-1
        state._subspace.L = 4
        state._subspace.get_dimension = lambda: 8
        state._subspace.idx_to_state = lambda x: [x+1]

        state._vec = np.zeros(8, dtype=np.complex128)
        state._vec[1] = 1
        state.set_initialized()

        self.assertEqual(
            str(state),
            '|0100>'
        )
        self.assertEqual(
            state._repr_latex_(),
            r'$\left|0100\right>$'
        )

    def test_large(self):
        state = State(L=7)
        state._vec = np.zeros(128, dtype=np.complex128)
        state._vec[:] = 0.001*(1+np.arange(128))
        state.set_initialized()

        self.assertEqual(
            str(state),
            '0.001|0000000> + 0.002|1000000> + 0.003|0100000> + ... + '
            '0.128|1111111>'
        )
        self.assertEqual(
            state._repr_latex_(),
            r'$0.001\left|0000000\right> + 0.002\left|1000000\right> + '
            r'0.003\left|0100000\right> + \cdots + 0.128\left|1111111\right>$'
        )

    def test_uninitialized(self):
        state = State(L=4)
        self.assertEqual(
            str(state),
            '<State with uninitialized contents>'
        )
        self.assertEqual(
            state._repr_latex_(),
            '<State with uninitialized contents>'
        )

    def test_zero(self):
        state = State(L=4)
        state.vec[:] = 0
        state.set_initialized()

        self.assertEqual(
            str(state),
            '<zero vector>'
        )
        self.assertEqual(
            state._repr_latex_(),
            '<zero vector>'
        )


if __name__ == '__main__':
    ut.main()
