# pylint: disable=W0212, W0104
'''
Unit tests for subspace.py.

These tests should NOT require MPI.
'''

import unittest as ut
from unittest.mock import Mock
import numpy as np
from dynamite.subspaces import Full, Parity, Explicit, Auto, SpinConserve
from dynamite.msc_tools import msc_dtype
from dynamite._backend.bsubspace import compute_rcm
from dynamite._backend.bbuild import dnm_int_t


class TestFull(ut.TestCase):

    def test_dimension(self):
        test_cases = [
            (2, 4),
            (10, 1024),
            (12, 4096)
        ]

        for L, dim in test_cases:
            with self.subTest(L=L):
                # do not call init, we're manually setting fields
                sp = object.__new__(Full)
                sp._L = L
                self.assertEqual(sp.get_dimension(), dim)

    def test_mapping_single(self):
        sp = object.__new__(Full)
        sp._L = 5
        self.assertEqual(sp.idx_to_state(np.array(10, dtype=dnm_int_t)), 10)
        self.assertEqual(sp.state_to_idx(np.array(10, dtype=dnm_int_t)), 10)

    def test_mapping_invalid_i2s(self):
        L = 5
        for idx in [-1, 2**L]:
            with self.subTest(idx=idx):
                with self.assertRaises(ValueError):
                    sp = object.__new__(Full)
                    sp._L = L
                    sp.idx_to_state(np.array([idx], dtype=dnm_int_t))

    def test_mapping_array(self):
        sp = object.__new__(Full)
        sp._L = 10
        ins = np.arange(2**sp._L)
        states = sp.idx_to_state(ins)
        idxs = sp.state_to_idx(ins)

        for i, state, idx in zip(ins, states, idxs):
            self.assertEqual(state, i)
            self.assertEqual(idx, i)

    def test_product_state_basis(self):
        self.assertTrue(Full._product_state_basis)

    def test_no_L(self):
        s = Full()
        with self.assertRaises(ValueError):
            s.get_dimension()

        with self.assertRaises(ValueError):
            s.idx_to_state(0)

        with self.assertRaises(ValueError):
            s.state_to_idx(0)

    def test_change_L(self):
        sp = Full()
        sp.L = 5
        with self.assertRaises(AttributeError):
            sp.L = 6


class TestParity(ut.TestCase):

    def test_dimension(self):
        test_cases = [
            (2, 2),
            (10, 512),
            (12, 2048)
        ]
        for L, dim in test_cases:
            for space in (0, 1):
                with self.subTest(L=L, parity=space):
                    sp = object.__new__(Parity)
                    sp._L = L
                    sp._space = space
                    self.assertEqual(sp.get_dimension(), dim)

    def test_product_state_basis(self):
        self.assertTrue(Parity._product_state_basis)

    def test_mapping_single(self):
        for idx, state_str in [(5, '01010'), (7, '01111')]:
            with self.subTest(idx=idx, state=state_str):
                sp = object.__new__(Parity)
                sp._L = 5
                sp._space = 0

                s = sp.idx_to_state(idx)
                self.assertEqual(s, int(state_str, 2))

                i = sp.state_to_idx(s)
                self.assertEqual(i, idx)

    def test_mapping_invalid_i2s(self):
        L = 5
        for idx in [-1, 2**(L-1)]:
            for parity in (0, 1):
                with self.subTest(idx=idx, parity=parity):
                    sp = object.__new__(Parity)
                    sp._L = L
                    sp._space = parity
                    with self.assertRaises(ValueError):
                        sp.idx_to_state(idx)

    def test_mapping_invalid_s2i(self):
        for space, state_str in [(0, '01011'), (1, '01010')]:
            with self.subTest(space=space, state=state_str):
                sp = object.__new__(Parity)
                sp._L = 5
                sp._space = space
                i = sp.state_to_idx(int(state_str, 2))
                self.assertEqual(i, -1)

    def test_mapping_array(self):
        L = 4

        correct_states = [
            [
                '0000',
                '0011',
                '0101',
                '0110',
                '1001',
                '1010',
                '1100',
                '1111',
            ],
            [
                '0001',
                '0010',
                '0100',
                '0111',
                '1000',
                '1011',
                '1101',
                '1110',
            ]
        ]

        bad_states = ['0111', '0110']

        for p in (0, 1):
            with self.subTest(parity=p):
                correct = np.array(
                    [int(x, 2) for x in correct_states[p]],
                    dtype=dnm_int_t
                )

                sp = object.__new__(Parity)
                sp._L = L
                sp._space = p

                states = sp.idx_to_state(np.arange(len(correct), dtype=dnm_int_t))
                idxs = sp.state_to_idx(correct)

                for s, c in zip(states, correct):
                    self.assertEqual(s, c)

                for i, idx in enumerate(idxs):
                    self.assertEqual(i, idx)

                correct[2] = int(bad_states[p], 2)
                idxs = sp.state_to_idx(correct)
                self.assertEqual(idxs[2], -1)

    def test_no_L(self):
        for p in ('even', 'odd'):
            with self.subTest(parity=p):
                s = Parity(p)
                with self.assertRaises(ValueError):
                    s.get_dimension()

                with self.assertRaises(ValueError):
                    s.idx_to_state(0)

                with self.assertRaises(ValueError):
                    s.state_to_idx(0)

    def test_change_L(self):
        for p in ('even', 'odd'):
            with self.subTest(parity=p):
                s = Parity(p)
                s.L = 5
                with self.assertRaises(AttributeError):
                    s.L = 6


class TestSpinConserve(ut.TestCase):

    def test_dimension(self):
        # each tuple is (L, k, dim)
        test_cases = [
            (2, 1, 2),
            (10, 2, 45),
            (10, 5, 252),
        ]
        for L, k, dim in test_cases:
            with self.subTest(L=L, k=k):
                sp = object.__new__(SpinConserve)
                sp._L = L
                sp._k = k
                sp._nchoosek = SpinConserve._compute_nchoosek(L, k)
                sp._spinflip = 0
                self.assertEqual(sp.get_dimension(), dim)

                if L == 2*k:
                    for spinflip in (-1, 1):
                        with self.subTest(spinflip=spinflip):
                            sp._spinflip = spinflip
                            self.assertEqual(sp.get_dimension(), dim//2)

    def test_product_state_basis_spinflip(self):
        for sign in '+-':
            with self.subTest(sign=sign):
                self.assertFalse(SpinConserve(4, 2, spinflip=sign).product_state_basis)

    def test_product_state_basis_no_spinflip(self):
        self.assertTrue(SpinConserve(4, 2).product_state_basis)

    def test_parameter_exceptions(self):
        # test cases are (L, k)
        test_cases = [
            (-1, 1),
            (5, -1),
            (5, 6),
        ]
        for L, k in test_cases:
            with self.subTest(L=L, k=k):
                with self.assertRaises(ValueError):
                    SpinConserve(L, k)

    def test_static_L(self):
        s = SpinConserve(10, 5)
        s.L = 10

        with self.assertRaises(AttributeError):
            s.L = 9

    def test_mapping_single(self):
        sp = object.__new__(SpinConserve)
        sp._L = 6
        sp._k = 3
        sp._nchoosek = SpinConserve._compute_nchoosek(sp.L, sp.k)
        sp._spinflip = 0

        s = sp.idx_to_state(5)
        self.assertEqual(s, int('010101', 2))

        i = sp.state_to_idx(s)
        self.assertEqual(i, 5)

    def test_mapping_invalid_s2i(self):
        sp = object.__new__(SpinConserve)
        sp._L = 5
        sp._k = 1
        sp._spinflip = 0
        sp._nchoosek = SpinConserve._compute_nchoosek(sp.L, sp.k)

        i = sp.state_to_idx(int('01011', 2))
        self.assertEqual(i, -1)

        sp._k = 2
        sp._nchoosek = SpinConserve._compute_nchoosek(sp.L, sp.k)
        i = sp.state_to_idx(int('01011', 2))
        self.assertEqual(i, -1)

        sp._k = 3
        sp._nchoosek = SpinConserve._compute_nchoosek(sp.L, sp.k)
        i = sp.state_to_idx(int('01010', 2))
        self.assertEqual(i, -1)

    def test_mapping_array(self):
        sp = object.__new__(SpinConserve)
        sp._spinflip = 0

        sp._L = 4
        ks = [1, 2]

        corrects = [
            np.array([
                0b0001,
                0b0010,
                0b0100,
                0b1000
            ], dtype=dnm_int_t),
            np.array([
                0b0011,
                0b0101,
                0b0110,
                0b1001,
                0b1010,
                0b1100,
            ], dtype=dnm_int_t),
        ]

        bad_state = 0b0111

        for k, correct in zip(ks, corrects):
            with self.subTest(k=k):
                sp._k = k
                sp._nchoosek = SpinConserve._compute_nchoosek(sp.L, sp.k)

                states = sp.idx_to_state(np.arange(len(correct), dtype=dnm_int_t))

                dim = sp.get_dimension()
                for idx in [-1, dim]:
                    with self.subTest(idx=idx):
                        with self.assertRaises(ValueError):
                            sp.idx_to_state(idx)

                idxs = sp.state_to_idx(correct)

                for s, c in zip(states, correct):
                    self.assertEqual(s, c)

                for i, idx in enumerate(idxs):
                    self.assertEqual(i, idx)

                correct[2] = bad_state
                idxs = sp.state_to_idx(correct)
                self.assertEqual(idxs[2], -1)

    def test_mapping_array_spinflip(self):
        sp = object.__new__(SpinConserve)

        sp._L = 4
        sp._k = sp.L//2
        sp._nchoosek = SpinConserve._compute_nchoosek(sp.L, sp.k)
        sp._spinflip = 1

        bad_state = 0b1001

        for spinflip in (-1, 1):
            with self.subTest(spinflip=spinflip):
                correct = np.array([
                    0b0011,
                    0b0101,
                    0b0110,
                ], dtype=dnm_int_t)
                space_size = len(correct)

                with self.assertRaises(ValueError):
                    sp.idx_to_state(np.arange(space_size, 2*space_size, dtype=dnm_int_t))

                idxs = sp.state_to_idx(correct)
                for i, idx in enumerate(idxs):
                    self.assertEqual(i, idx)

                correct[2] = bad_state
                idxs = sp.state_to_idx(correct)
                self.assertEqual(idxs[2], -1)

    def test_spinflip_exception(self):
        for sign in '+-':
            with self.subTest(sign=sign):
                with self.assertRaises(ValueError):
                    SpinConserve(5, 2, spinflip=sign)

    def test_compare_to_auto(self):
        test_cases = [
            'DDUUUUUU',
            'DDDDUUUU',
        ]

        from dynamite.operators import sigmax, sigmay, index_sum
        H = index_sum(sigmax(0)*sigmax(1) + sigmay(0)*sigmay(1), size=8)

        for state in test_cases:
            with self.subTest(state=state):
                sp1 = Auto(H, state, sort=True)
                sp2 = SpinConserve(len(state), state.count('D'))
                self.assertEqual(sp1, sp2)

    @classmethod
    def msc_from_array(cls, ary):
        msc = [(int(m, 2), int(s, 2), c) for m, s, c in ary]
        return np.array(msc, copy=False, dtype=msc_dtype)

    def check_reduce_msc_equal(self, initial, correct, spinflip, L, conserves):
        sp = object.__new__(SpinConserve)
        sp._spinflip = spinflip
        sp._L = L

        check_msc, conserved = sp.reduce_msc(self.msc_from_array(initial), check_conserves=True)
        correct_msc = self.msc_from_array(correct)

        self.assertTrue(
            np.array_equal(check_msc, correct_msc),
            msg=f'\n{check_msc}\n\n{correct_msc}\n'
        )
        self.assertEqual(conserved, conserves)

    def test_reduce_msc_flip_sign(self):
        initial = [
            ('0011', '0000', 0.75),
            ('1100', '0000', 0.25),
        ]

        for spinflip in (1, -1):
            with self.subTest(spinflip=spinflip):
                correct = [
                    ('0011', '0000', 0.75+spinflip*0.25),
                ]

                self.check_reduce_msc_equal(
                    initial, correct, spinflip, 4, True
                )

    def test_reduce_msc_flip_sign_cancel(self):
        initial = [
            ('0011', '0000', 1),
            ('1100', '0000', 1),
        ]

        for spinflip in (1, -1):
            with self.subTest(spinflip=spinflip):
                if spinflip == 1:
                    correct = [
                        ('0011', '0000', 2),
                    ]
                else:
                    correct = [
                    ]

                self.check_reduce_msc_equal(
                    initial, correct, spinflip, 4, True
                )

    def test_reduce_msc_flip_odd_Z(self):
        initial = [
            ('0000', '0001', 1),
            ('0000', '1101', 1),
            ('0000', '1110', 1),
        ]

        for spinflip in (1, -1):
            with self.subTest(spinflip=spinflip):
                correct = [
                ]

                self.check_reduce_msc_equal(
                    initial, correct, spinflip, 4, False
                )

    def test_reduce_msc_flip_even_Z(self):
        initial = [
            ('0000', '0011', 1),
            ('0000', '1001', 1),
            ('0000', '1111', 1),
        ]

        for spinflip in (1, -1):
            with self.subTest(spinflip=spinflip):
                correct = [
                    ('0000', '0011', 1),
                    ('0000', '1001', 1),
                    ('0000', '1111', 1),
                ]

                self.check_reduce_msc_equal(
                    initial, correct, spinflip, 4, True
                )

    def test_reduce_msc_flip_XZ(self):
        initial = [
            ('1100', '0011', 1),
        ]

        for spinflip in (1, -1):
            with self.subTest(spinflip=spinflip):
                correct = [
                    ('0011', '0011', spinflip),
                ]

                self.check_reduce_msc_equal(
                    initial, correct, spinflip, 4, True
                )

    def test_reduce_msc_flip_XYZ(self):
        initial = [
            ('0011', '0011', 0.75),
            ('1100', '0011', 0.25),
        ]

        for spinflip in (1, -1):
            with self.subTest(spinflip=spinflip):
                correct = [
                    ('0011', '0011', 0.75+spinflip*0.25),
                ]

                self.check_reduce_msc_equal(
                    initial, correct, spinflip, 4, True
                )


class TestRCM(ut.TestCase):

    def setUp(self):
        msc = [
            ('0011', '0000', 1),
            ('0011', '0011', -1),
            ('0110', '0000', 1),
            ('0110', '0110', -1),
            ('1100', '0000', 1),
            ('1100', '1100', -1),
        ]

        self.masks = np.array([int(x[0], 2) for x in msc], dtype=dnm_int_t)
        self.signs = np.array([int(x[1], 2) for x in msc], dtype=dnm_int_t)
        self.coeffs = np.array([x[2] for x in msc], dtype=np.complex128)

    def test_half(self):
        state_map = np.ndarray((16,), dtype=dnm_int_t)

        dim = compute_rcm(self.masks, self.signs, self.coeffs, state_map, 3, 4)

        self.assertEqual(dim, 6)

    def test_trimmed(self):
        state_map = np.ndarray((6,), dtype=dnm_int_t)

        dim = compute_rcm(self.masks, self.signs, self.coeffs, state_map, 3, 4)

        self.assertEqual(dim, 6)

    def test_too_short(self):
        state_map = np.ndarray((4,), dtype=dnm_int_t)
        with self.assertRaises(RuntimeError):
            compute_rcm(self.masks, self.signs, self.coeffs, state_map, 3, 4)

    def test_one(self):
        state_map = np.ndarray((16,), dtype=dnm_int_t)

        dim = compute_rcm(self.masks, self.signs, self.coeffs, state_map, 1, 4)

        self.assertEqual(dim, 4)

    def test_zero(self):
        state_map = np.ndarray((16,), dtype=dnm_int_t)

        dim = compute_rcm(self.masks, self.signs, self.coeffs, state_map, 0, 4)

        self.assertEqual(dim, 1)


class TestExplicit(ut.TestCase):

    def setUp(self):
        self.test_states = {
            'unsorted': [
                0b01110,
                0b11110,
                0b10101,
                0b10011,
                0b00011,
                0b01101,
                0b00101,
                0b11101,
            ]
        }
        self.test_states['sorted'] = sorted(
            self.test_states['unsorted']
        )

    def test_L_error(self):
        for name, states in self.test_states.items():
            with self.subTest(which=name):
                s = Explicit(states)
                with self.assertRaises(ValueError):
                    s.L = 4

    def test_dimension(self):
        for name, states in self.test_states.items():
            with self.subTest(which=name):
                s = Explicit(states)
                s.L = 5
                self.assertEqual(s.get_dimension(), len(states))

    def test_s2i(self):
        for name, states in self.test_states.items():
            with self.subTest(which=name):
                s = Explicit(states)
                s.L = 5

                for state in states:
                    with self.subTest(state=bin(state)[2:]):
                        self.assertEqual(s.state_to_idx(state),
                                         states.index(state))

                # an array of states
                self.assertTrue(np.all(
                    s.state_to_idx(states) == np.arange(len(states))
                ))

                # a state not in the list
                bad_state = 0b11111
                self.assertEqual(s.state_to_idx(bad_state), -1)

    def test_i2s(self):
        for name, states in self.test_states.items():
            with self.subTest(which=name):
                s = Explicit(states)
                s.L = 5

                for idx in range(len(states)):
                    with self.subTest(idx=idx):
                        self.assertEqual(s.idx_to_state(idx),
                                         states[idx])

                # an array of states
                self.assertTrue(np.all(
                    s.idx_to_state(np.arange(len(states))) == np.array(states)
                ))
                for idx in [-1, len(states)]:
                    with self.subTest(idx=idx):
                        with self.assertRaises(ValueError):
                            s.idx_to_state(idx)

    def test_compare_parity(self):
        p = Parity('even')
        p.L = 5

        s = Explicit(p.idx_to_state(np.arange(p.get_dimension())))
        s.L = 5

        self.assertEqual(p, s)

    def test_compare_spinconserve(self):
        p = SpinConserve(L=6, k=3)

        s = Explicit(p.idx_to_state(np.arange(p.get_dimension())))
        s.L = 6

        self.assertEqual(p, s)


class TestAuto(ut.TestCase):

    @classmethod
    def to_bin(cls, x, L):
        return bin(x)[2:].zfill(L)

    def test_string_int_state(self):
        from dynamite.operators import sigmax, sigmay, index_sum
        H = index_sum(sigmax(0)*sigmax(1) + sigmay(0)*sigmay(1), size=8)
        sp1 = Auto(H,     'DDUUUUUU')
        sp2 = Auto(H, int('00000011',2))
        sp3 = Auto(H, int('11000000',2))

        self.assertEqual(sp1.state, sp2.state)
        self.assertNotEqual(sp1.state, sp3.state)
        self.assertNotEqual(sp2.state, sp3.state)

        self.assertEqual(sp1, sp2)
        self.assertEqual(sp1, sp3)

    def test_XX_parity(self):
        '''
        enforce that Auto on XX operator gives the same result as the Parity
        subspace
        '''
        from dynamite.operators import sigmax, index_sum
        H = index_sum(sigmax(0)*sigmax(1), size=8)
        auto = Auto(H, 'U'*8)
        parity = Parity('even')
        parity.L = 8

        sorted_parity = np.sort(parity.idx_to_state(np.arange(parity.get_dimension())))

        msg = ''
        if auto.get_dimension() != parity.get_dimension():
            msg += 'dimensions differ. auto dim: %d, parity dim: %d' % \
                (auto.get_dimension(), parity.get_dimension())
        else:
            msg += '\nauto:     parity:\n'
            for i in range(auto.get_dimension()):
                msg += self.to_bin(auto.state_map[i], 8)
                msg += '  '
                msg += self.to_bin(sorted_parity[i], 8)
                msg += '\n'

        self.assertTrue(
            auto.get_dimension() == parity.get_dimension(),
            msg=msg
        )

        self.assertTrue(
            np.all(auto.state_map == sorted_parity),
            msg=msg
        )


class Checksum(ut.TestCase):

    def test_same_full(self):
        space = Full()
        space.L = 10
        chksum1 = space.get_checksum()
        chksum2 = space.copy().get_checksum()
        self.assertEqual(chksum1, chksum2)

    def test_diff_full(self):
        space1 = Full()
        space1.L = 10

        space2 = Full()
        space2.L = 11

        self.assertNotEqual(
            space1.get_checksum(),
            space2.get_checksum()
        )

    def test_same_full_large(self):
        space = Full()
        space.L = 20
        chksum1 = space.get_checksum()
        chksum2 = space.copy().get_checksum()
        self.assertEqual(chksum1, chksum2)

    def test_diff_full_large(self):
        space1 = Full()
        space1.L = 20

        space2 = Full()
        space2.L = 21

        self.assertNotEqual(
            space1.get_checksum(),
            space2.get_checksum()
        )

    def test_same_parity(self):
        space = Parity('even')
        space.L = 10
        chksum1 = space.get_checksum()
        chksum2 = space.get_checksum()
        self.assertEqual(chksum1, chksum2)

    def test_diff_parity(self):
        space0 = Parity('even')
        space1 = Parity('odd')
        space0.L = 10
        space1.L = 10

        chksum0 = space0.get_checksum()
        chksum1 = space1.get_checksum()

        self.assertNotEqual(chksum0, chksum1)


if __name__ == '__main__':
    ut.main()
