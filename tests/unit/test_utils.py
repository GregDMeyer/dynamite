# pylint: disable=W0212
'''
Unit tests for operators.py.

These tests should NOT require MPI.
'''

import unittest as ut

from dynamite._utils import popcount, parity

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

    def test_parity(self):
        for x,p in self.test_cases:
            with self.subTest(x=x):
                self.assertEqual(parity(x), p%2)

if __name__ == '__main__':
    ut.main()
