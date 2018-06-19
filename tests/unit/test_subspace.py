# pylint: disable=W0212, W0104
'''
Unit tests for subspace.py.

These tests should NOT require MPI.
'''

# TODO: change dtypes to correct values

import unittest as ut
import numpy as np
from dynamite.subspace import Full, Parity, Auto

class TestFull(ut.TestCase):

    def test_dimension(self):
        test_cases = [
            (2, 4),
            (10, 1024),
            (12, 4096)
        ]
        for L, dim in test_cases:
            with self.subTest(L = L):
                self.assertEqual(Full._get_dimension(L, 0), dim)

    def test_mapping_single(self):
        self.assertEqual(Full._idx_to_state(np.array(10, dtype=np.int32), 5, None), 10)
        self.assertEqual(Full._state_to_idx(np.array(10, dtype=np.int32), 5, None), 10)

        self.assertEqual(Full._idx_to_state(np.array(32, dtype=np.int32), 5, None), -1)
        self.assertEqual(Full._state_to_idx(np.array(32, dtype=np.int32), 5, None), -1)

    def test_mapping_array(self):
        L = 10
        ins = np.arange(2**L)
        ins[10] = 1024
        ins[500] = 1024
        states = Full._idx_to_state(ins, L, None)
        idxs = Full._state_to_idx(ins, L, None)

        for i, state, idx in zip(ins, states, idxs):
            if i != 1024:
                self.assertEqual(state, i)
                self.assertEqual(idx, i)
            else:
                self.assertEqual(state, -1)
                self.assertEqual(idx, -1)

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

    def test_mapping_invalid_i2s(self):
        s = Parity._idx_to_state(np.array([16], dtype=np.int32), 5, 0)
        self.assertEqual(s, -1)

        s = Parity._idx_to_state(np.array([16], dtype=np.int32), 5, 1)
        self.assertEqual(s, -1)

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

        bad_states = [
            ['0111', '10010'],
            ['0110', '11010']
        ]

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

                idxs[2] = 8
                self.assertEqual(Parity._idx_to_state(idxs, L, p)[2], -1)

                correct[2] = int(bad_states[p][0], 2)
                correct[5] = int(bad_states[p][1], 2)
                idxs = Parity._state_to_idx(correct, L, p)
                self.assertEqual(idxs[2], -1)
                self.assertEqual(idxs[5], -1)

class TestAuto(ut.TestCase):

    def test_implement(self):
        raise NotImplementedError()

if __name__ == '__main__':
    ut.main()
