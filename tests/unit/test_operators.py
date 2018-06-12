# pylint: disable=W0212
'''
Unit tests for operators.py.

These tests should NOT require MPI.
'''

import unittest as ut
import numpy as np
# from unittest.mock import Mock, MagicMock
#
# import mock_backend

"""
TESTS TO WRITE:
 - save method
  - use mock_open
 - build_mat: ensuring diagonal entries
 - shift_index logic
"""

from dynamite.operators import Operator

class ToNumpy(ut.TestCase):
    '''
    Test the Operator._MSC_to_numpy method, whose behavior defines the MSC array.
    So in some sense these tests are the definition of the MSC array.
    '''

    def check_same(self, dnm, npy):
        '''
        Helper function to check that dynamite and numpy arrays are equal, and
        print the differences if not.
        '''
        self.assertTrue(np.all(dnm == npy),
                        msg = '\n\n'.join(['\ndnm:\n'+str(dnm), '\nnpy:\n'+str(npy)]))

    def test_identity(self):
        dnm = Operator._MSC_to_numpy([(0, 0, 1)], (5,5))
        npy = np.identity(5)
        self.check_same(dnm, npy)

    def test_identity_wide(self):
        dnm = Operator._MSC_to_numpy([(0, 0, 1)], (3,5),
                                     idx_to_state = lambda x: x if x < 3 else None)
        npy = np.identity(5)[:3,:]
        self.check_same(dnm, npy)

    def test_identity_tall(self):
        dnm = Operator._MSC_to_numpy([(0, 0, 1)], (5,3),
                                     state_to_idx = lambda x: x if x < 3 else None)
        npy = np.identity(5)[:,:3]
        self.check_same(dnm, npy)

    def test_allflip(self):
        dnm = Operator._MSC_to_numpy([(15, 0, 1)], (16,16))
        npy = np.identity(16)[:,::-1]
        self.check_same(dnm, npy)

    def test_sign1(self):
        dnm = Operator._MSC_to_numpy([(0, 1, 1)], (16,16))
        npy = np.diag([1, -1]*8)
        self.check_same(dnm, npy)

    def test_sign2(self):
        dnm = Operator._MSC_to_numpy([(0, 3, 1)], (16,16))
        npy = np.diag([1, -1, -1, 1]*4)
        self.check_same(dnm, npy)

    def test_signL(self):
        dnm = Operator._MSC_to_numpy([(0, 8, 1)], (16,16))
        npy = np.diag([1]*8 + [-1]*8)
        self.check_same(dnm, npy)

    def test_signL2(self):
        dnm = Operator._MSC_to_numpy([(0, 9, 1)], (16,16))
        npy = np.diag([1, -1]*4 + [-1, 1]*4)
        self.check_same(dnm, npy)

    def test_full(self):
        dnm = Operator._MSC_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8))
        npy = np.array(
            [
                [    0,-0.5j,    0,    0,   -2,    0,    0,    0 ],
                [ 0.5j,    0,    0,    0,    0,    2,    0,    0 ],
                [    0,    0,    0,-0.5j,    0,    0,    2,    0 ],
                [    0,    0, 0.5j,    0,    0,    0,    0,   -2 ],
                [   -2,    0,    0,    0,    0, 0.5j,    0,    0 ],
                [    0,    2,    0,    0,-0.5j,    0,    0,    0 ],
                [    0,    0,    2,    0,    0,    0,    0, 0.5j ],
                [    0,    0,    0,   -2,    0,    0,-0.5j,    0 ],
            ]
        )
        self.check_same(dnm, npy)

    def test_map1(self):
        dnm = Operator._MSC_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
                                     idx_to_state = lambda x: x^4)
        npy = np.array(
            [
                [   -2,    0,    0,    0,    0, 0.5j,    0,    0 ],
                [    0,    2,    0,    0,-0.5j,    0,    0,    0 ],
                [    0,    0,    2,    0,    0,    0,    0, 0.5j ],
                [    0,    0,    0,   -2,    0,    0,-0.5j,    0 ],
                [    0,-0.5j,    0,    0,   -2,    0,    0,    0 ],
                [ 0.5j,    0,    0,    0,    0,    2,    0,    0 ],
                [    0,    0,    0,-0.5j,    0,    0,    2,    0 ],
                [    0,    0, 0.5j,    0,    0,    0,    0,   -2 ],
            ]
        )
        self.check_same(dnm, npy)

    def test_map2(self):
        dnm = Operator._MSC_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
                                     state_to_idx = lambda x: x^4)
        npy = np.array(
            [
                [   -2,    0,    0,    0,    0,-0.5j,    0,    0 ],
                [    0,    2,    0,    0, 0.5j,    0,    0,    0 ],
                [    0,    0,    2,    0,    0,    0,    0,-0.5j ],
                [    0,    0,    0,   -2,    0,    0, 0.5j,    0 ],
                [    0, 0.5j,    0,    0,   -2,    0,    0,    0 ],
                [-0.5j,    0,    0,    0,    0,    2,    0,    0 ],
                [    0,    0,    0, 0.5j,    0,    0,    2,    0 ],
                [    0,    0,-0.5j,    0,    0,    0,    0,   -2 ],
            ]
        )
        self.check_same(dnm, npy)

    def test_map_both(self):
        dnm = Operator._MSC_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
                                     state_to_idx = lambda x: x^4,
                                     idx_to_state = lambda x: x^2)
        npy = np.array(
            [
                [    0,    0,    2,    0,    0,    0,    0,-0.5j ],
                [    0,    0,    0,   -2,    0,    0, 0.5j,    0 ],
                [   -2,    0,    0,    0,    0,-0.5j,    0,    0 ],
                [    0,    2,    0,    0, 0.5j,    0,    0,    0 ],
                [    0,    0,    0, 0.5j,    0,    0,    2,    0 ],
                [    0,    0,-0.5j,    0,    0,    0,    0,   -2 ],
                [    0, 0.5j,    0,    0,   -2,    0,    0,    0 ],
                [-0.5j,    0,    0,    0,    0,    2,    0,    0 ],
            ]
        )
        self.check_same(dnm, npy)

class Serialization(ut.TestCase):
    '''
    Test the Operator._serialize and Operator._deserialize methods.
    '''

    def setUp(self):
        dtype32 = np.dtype([('masks', np.int32),
                            ('signs', np.int32),
                            ('coeffs', np.complex128)])
        dtype64 = np.dtype([('masks', np.int64),
                            ('signs', np.int64),
                            ('coeffs', np.complex128)])

        self.test_cases = [
            {
                'L' : 20,
                'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                                 dtype = dtype32.newbyteorder('B')),
                'serial' : b'20\n2\n32\n' + \
                           b'\x00\x00\x00\x01\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x05\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            },
            {
                'L' : 20,
                'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                                 dtype = dtype32.newbyteorder('L')),
                'serial' : b'20\n2\n32\n' + \
                           b'\x00\x00\x00\x01\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x05\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            },
            {
                'L' : 20,
                'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                                 dtype = dtype64.newbyteorder('B')),
                'serial' : b'20\n2\n64\n' + \
                           b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            },
            {
                'L' : 20,
                'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                                 dtype = dtype64.newbyteorder('L')),
                'serial' : b'20\n2\n64\n' + \
                           b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            }
        ]

    def test_serialize(self):
        for n, case in enumerate(self.test_cases):
            with self.subTest(n = n):
                ser = Operator._serialize(case['L'], case['MSC'])
                self.assertEqual(ser, case['serial'])

    def test_deserialize(self):
        for n, case in enumerate(self.test_cases):
            with self.subTest(n = n):
                L, MSC = Operator._deserialize(case['serial'])
                self.assertEqual(L, case['L'])
                self.assertTrue(np.all(MSC == case['MSC']),
                                msg = '\n'+'\n\n'.join([str(MSC), str(case['MSC'])]))

if __name__ == '__main__':
    ut.main()
