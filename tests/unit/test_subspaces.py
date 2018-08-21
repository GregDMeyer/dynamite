# pylint: disable=W0212, W0104
'''
Unit tests for subspace.py.

These tests should NOT require MPI.
'''

# TODO: change dtypes to correct values

import unittest as ut
import numpy as np
from dynamite.subspaces import Full, Parity
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
        self.assertEqual(Full._idx_to_state(np.array(10, dtype=np.int32), 5), 10)
        self.assertEqual(Full._state_to_idx(np.array(10, dtype=np.int32), 5), 10)

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
        s = Parity._idx_to_state(np.array([5], dtype=np.int32), 5, 0)
        self.assertEqual(s, int('00101', 2))

        i = Parity._state_to_idx(s, 5, 0)
        self.assertEqual(i, 5)

        s = Parity._idx_to_state(np.array([7], dtype=np.int32), 5, 0)
        self.assertEqual(s, int('10111', 2))

        i = Parity._state_to_idx(s, 5, 0)
        self.assertEqual(i, 7)

    def test_mapping_invalid_s2i(self):
        i = Parity._state_to_idx(np.array([int('01011',2)], dtype=np.int32), 5, 0)
        self.assertEqual(i, -1)

        i = Parity._state_to_idx(np.array([int('01010',2)], dtype=np.int32), 5, 1)
        self.assertEqual(i, -1)

    def test_mapping_array(self):
        L = 4

        correct_states = [
            [
                '0000',
                '1001',
                '1010',
                '0011',
                '1100',
                '0101',
                '0110',
                '1111',
            ],
            [
                '1000',
                '0001',
                '0010',
                '1011',
                '0100',
                '1101',
                '1110',
                '0111',
            ]
        ]

        bad_states = ['0111', '0110']

        for p in (0, 1):
            with self.subTest(parity = p):
                correct = np.array(
                    [int(x,2) for x in correct_states[p]],
                    dtype = np.int32
                )

                states = Parity._idx_to_state(np.arange(len(correct), dtype=np.int32), L, p)
                idxs = Parity._state_to_idx(correct, L, p)

                for s, c in zip(states, correct):
                    self.assertEqual(s, c)

                for i, idx in enumerate(idxs):
                    self.assertEqual(i, idx)

                correct[2] = int(bad_states[p], 2)
                idxs = Parity._state_to_idx(correct, L, p)
                self.assertEqual(idxs[2], -1)

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
        state_rmap = np.ndarray((16,), dtype=dnm_int_t)
        state_rmap[:] = -1

        dim = compute_rcm(self.masks, self.signs, self.coeffs, state_map, state_rmap, 3, 4)

        self.assertEqual(dim, 6)
        for i in range(dim):
            self.assertEqual(state_rmap[state_map[i]], i)

    def test_trimmed(self):
        state_map = np.ndarray((6,), dtype=dnm_int_t)
        state_rmap = np.ndarray((16,), dtype=dnm_int_t)
        state_rmap[:] = -1

        dim = compute_rcm(self.masks, self.signs, self.coeffs, state_map, state_rmap, 3, 4)

        self.assertEqual(dim, 6)
        for i in range(dim):
            self.assertEqual(state_rmap[state_map[i]], i)

    def test_too_short(self):
        state_map = np.ndarray((4,), dtype=dnm_int_t)
        state_rmap = np.ndarray((16,), dtype=dnm_int_t)
        state_rmap[:] = -1

        with self.assertRaises(RuntimeError):
            compute_rcm(self.masks, self.signs, self.coeffs, state_map, state_rmap, 3, 4)

    def test_one(self):
        state_map = np.ndarray((16,), dtype=dnm_int_t)
        state_rmap = np.ndarray((16,), dtype=dnm_int_t)
        state_rmap[:] = -1

        dim = compute_rcm(self.masks, self.signs, self.coeffs, state_map, state_rmap, 1, 4)

        self.assertEqual(dim, 4)
        for i in range(dim):
            self.assertEqual(state_rmap[state_map[i]], i)

    def test_zero(self):
        state_map = np.ndarray((16,), dtype=dnm_int_t)
        state_rmap = np.ndarray((16,), dtype=dnm_int_t)
        state_rmap[:] = -1

        dim = compute_rcm(self.masks, self.signs, self.coeffs, state_map, state_rmap, 0, 4)

        self.assertEqual(dim, 1)
        for i in range(dim):
            self.assertEqual(state_rmap[state_map[i]], i)

class TestAuto(ut.TestCase):
    # TODO
    pass

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
