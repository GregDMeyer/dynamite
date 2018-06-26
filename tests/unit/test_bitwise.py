# pylint: disable=W0212
'''
Unit tests for operators.py.

These tests should NOT require MPI.
'''

import unittest as ut
import numpy as np

from dynamite.bitwise import popcount, parity, intlog2

class PopcountParity(ut.TestCase):

    test_cases = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 1),
        (6, 2),
        (9, 2),
        (11, 3),
        (12, 2),
        (35, 3),
        (59, 5),
        (148742, 6)
    ]

    def test_popcount(self):
        for x,p in self.test_cases:
            with self.subTest(x=x):
                self.assertEqual(popcount(x), p)

    def test_parity_single(self):
        for x,p in self.test_cases:
            with self.subTest(x=x):
                self.assertEqual(parity(x), p%2)

    def test_parity_array(self):
        x, p = np.array(self.test_cases, dtype = int).T
        self.assertTrue(np.all(parity(x) == p%2))

class IntLog2(ut.TestCase):

    test_cases = [
        (0, -1),
        (1, 0),
        (4, 2),
        (6, 2),
        (12, 3),
        (148742, 17)
    ]

    def test_single(self):
        for x,l in self.test_cases:
            with self.subTest(x=x):
                self.assertEqual(intlog2(x), l)

    def test_array(self):
        x, l = np.array(self.test_cases, dtype = int).T
        self.assertTrue(np.all(intlog2(x) == l))

if __name__ == '__main__':
    ut.main()
