# pylint: disable=W0212, W0104
'''
Unit tests for subspace.py.

These tests should NOT require MPI.
'''

import unittest as ut
import numpy as np
from dynamite.subspaces import Full, Parity, Auto
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
            with self.subTest(L = L):
                self.assertEqual(Full._get_dimension(L), dim)

    def test_mapping_single(self):
        self.assertEqual(Full._idx_to_state(np.array(10, dtype=dnm_int_t), 5), 10)
        self.assertEqual(Full._state_to_idx(np.array(10, dtype=dnm_int_t), 5), 10)

    def test_mapping_array(self):
        L = 10
        ins = np.arange(2**L)
        states = Full._idx_to_state(ins, L)
        idxs = Full._state_to_idx(ins, L)

        for i, state, idx in zip(ins, states, idxs):
            self.assertEqual(state, i)
            self.assertEqual(idx, i)

class TestParity(ut.TestCase):

    def test_dimension(self):
        test_cases = [
            (2, 2),
            (10, 512),
            (12, 2048)
        ]
        for L, dim in test_cases:
            with self.subTest(L = L):
                self.assertEqual(Parity._get_dimension(L, 0), dim)
                self.assertEqual(Parity._get_dimension(L, 1), dim)

    def test_mapping_single(self):
        s = Parity._idx_to_state(np.array([5], dtype=dnm_int_t), 5, 0)
        self.assertEqual(s, int('01010', 2))

        i = Parity._state_to_idx(s, 10, 0)
        self.assertEqual(i, 5)

        s = Parity._idx_to_state(np.array([7], dtype=dnm_int_t), 5, 0)
        self.assertEqual(s, int('01111', 2))

        i = Parity._state_to_idx(s, 15, 0)
        self.assertEqual(i, 7)

    def test_mapping_invalid_s2i(self):
        i = Parity._state_to_idx(np.array([int('01011',2)], dtype=dnm_int_t), 5, 0)
        self.assertEqual(i, -1)

        i = Parity._state_to_idx(np.array([int('01010',2)], dtype=dnm_int_t), 5, 1)
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
            with self.subTest(parity = p):
                correct = np.array(
                    [int(x,2) for x in correct_states[p]],
                    dtype = dnm_int_t
                )

                states = Parity._idx_to_state(np.arange(len(correct), dtype=dnm_int_t), L, p)
                idxs = Parity._state_to_idx(correct, L, p)

                for s, c in zip(states, correct):
                    self.assertEqual(s, c)

                for i, idx in enumerate(idxs):
                    self.assertEqual(i, idx)

                correct[2] = int(bad_states[p], 2)
                idxs = Parity._state_to_idx(correct, L, p)
                self.assertEqual(idxs[2], -1)

# TODO: see if we can enfore state_map being correct
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
        space = Full()
        space.L = 10
        chksum1 = space.get_checksum()

        space.L = 11
        chksum2 = space.get_checksum()

        self.assertNotEqual(chksum1, chksum2)

    def test_same_full_large(self):
        space = Full()
        space.L = 20
        chksum1 = space.get_checksum()
        chksum2 = space.copy().get_checksum()
        self.assertEqual(chksum1, chksum2)

    def test_diff_full_large(self):
        space = Full()
        space.L = 20
        chksum1 = space.get_checksum()

        space.L = 21
        chksum2 = space.get_checksum()

        self.assertNotEqual(chksum1, chksum2)

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

    # TODO: Auto tests?

if __name__ == '__main__':
    ut.main()
