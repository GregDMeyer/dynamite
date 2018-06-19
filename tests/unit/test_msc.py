# pylint: disable=W0212
'''
Unit tests for the classmethods of the Operator class in operators.py.

These tests should NOT require MPI.
'''

import unittest as ut
import numpy as np

from dynamite import msc

class ToNumpy(ut.TestCase):
    '''
    Test the msc.MSC_to_numpy method, whose behavior defines the MSC array.
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
        dnm = msc.MSC_to_numpy([(0, 0, 1)], (5,5))
        npy = np.identity(5)
        self.check_same(dnm, npy)

    def test_identity_wide(self):
        dnm = msc.MSC_to_numpy([(0, 0, 1)], (3,5),
                               idx_to_state = lambda x: x if x < 3 else None)
        npy = np.identity(5)[:3,:]
        self.check_same(dnm, npy)

    def test_identity_tall(self):
        dnm = msc.MSC_to_numpy([(0, 0, 1)], (5,3),
                               state_to_idx = lambda x: x if x < 3 else None)
        npy = np.identity(5)[:,:3]
        self.check_same(dnm, npy)

    def test_allflip(self):
        dnm = msc.MSC_to_numpy([(15, 0, 1)], (16,16))
        npy = np.identity(16)[:,::-1]
        self.check_same(dnm, npy)

    def test_sign1(self):
        dnm = msc.MSC_to_numpy([(0, 1, 1)], (16,16))
        npy = np.diag([1, -1]*8)
        self.check_same(dnm, npy)

    def test_sign2(self):
        dnm = msc.MSC_to_numpy([(0, 3, 1)], (16,16))
        npy = np.diag([1, -1, -1, 1]*4)
        self.check_same(dnm, npy)

    def test_signL(self):
        dnm = msc.MSC_to_numpy([(0, 8, 1)], (16,16))
        npy = np.diag([1]*8 + [-1]*8)
        self.check_same(dnm, npy)

    def test_signL2(self):
        dnm = msc.MSC_to_numpy([(0, 9, 1)], (16,16))
        npy = np.diag([1, -1]*4 + [-1, 1]*4)
        self.check_same(dnm, npy)

    def test_full(self):
        dnm = msc.MSC_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8))
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
        dnm = msc.MSC_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
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
        dnm = msc.MSC_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
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
        dnm = msc.MSC_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
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

class MSCSum(ut.TestCase):
    '''
    Test the _MSC_sum method.
    '''

    def setUp(self):
        self.dtype = np.dtype([('masks', np.int32),
                               ('signs', np.int32),
                               ('coeffs', np.complex128)])

    def check_same(self, check, target):
        check = msc.combine_and_sort(check)
        target = msc.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_single(self):
        check = msc.MSC_sum(np.array([(1, 2, 3)], dtype=self.dtype))
        target = [(1, 2, 3)]
        self.check_same(check, target)

    def test_2x1(self):
        lst = [
            np.array([(3, 2, 1j), (5, 6, 2)], dtype=self.dtype),
            np.array([(1, 2, 3)], dtype=self.dtype)
        ]
        check = msc.MSC_sum(lst)
        target = [(3, 2, 1j), (5, 6, 2), (1, 2, 3)]
        self.check_same(check, target)

    def test_iterable(self):
        check = msc.MSC_sum(np.array([(1, 2, i**2)], dtype=self.dtype) for i in range(5))
        target = [(1, 2, 30)]
        self.check_same(check, target)

    def test_empty(self):
        check = msc.MSC_sum([])
        target = []
        self.check_same(check, target)

class MSCProduct(ut.TestCase):
    '''
    Test the _MSC_product method.
    '''

    def setUp(self):
        self.dtype = np.dtype([('masks', np.int32),
                               ('signs', np.int32),
                               ('coeffs', np.complex128)])

    def check_same(self, check, target):
        check = msc.combine_and_sort(check)
        target = msc.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_single(self):
        check = msc.MSC_product(np.array([(1,2,3)], dtype=self.dtype))
        target = [(1,2,3)]
        self.check_same(check, target)

    def test_XX(self):
        lst = [
            [(1, 0, 2)],
            [(2, 0, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc.MSC_product(lst)
        target = [(3, 0, 6)]
        self.check_same(check, target)

    def test_ZZ(self):
        lst = [
            [(0, 1, 2)],
            [(0, 2, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc.MSC_product(lst)
        target = [(0, 3, 6)]
        self.check_same(check, target)

    def test_YY(self):
        lst = [
            [(1, 1, 2)],
            [(2, 2, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc.MSC_product(lst)
        target = [(3, 3, 6)]
        self.check_same(check, target)

    def test_XZ(self):
        lst = [
            [(1, 0, 2)],
            [(0, 1, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc.MSC_product(lst)
        target = [(1, 1, 6)]
        self.check_same(check, target)

    def test_ZX(self):
        lst = [
            [(0, 1, 2)],
            [(1, 0, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc.MSC_product(lst)
        target = [(1, 1, -6)]
        self.check_same(check, target)

    def test_1x2x3(self):
        lst = [
            [(1, 0, 1)],
            [(0, 1, 2), (3, 3, 5)],
            [(1, 0, 3), (6, 4, 7), (3, 4, 11)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc.MSC_product(lst)
        target = [(0, 1, -6),
                  (7, 5, 14),
                  (2, 5,-22),
                  (3, 3,-15),
                  (4, 7,-35),
                  (1, 7, 55)]
        self.check_same(check, target)

class ShiftMSC(ut.TestCase):
    '''
    Tests the shift method.
    '''

    def setUp(self):
        self.dtype = np.dtype([('masks', np.int32),
                               ('signs', np.int32),
                               ('coeffs', np.complex128)])

    def test_single_mask(self):
        MSC = np.array([(1, 0, 0.5j)], dtype = self.dtype)
        for i in range(5):
            with self.subTest(shift=i):
                shifted = msc.shift(MSC, i, None)
                self.assertEqual(shifted['masks'], 2**i)
                self.assertEqual(shifted['signs'], 0)
                self.assertEqual(shifted['coeffs'], 0.5j)
                # check that we haven't changed it
                self.assertTrue(np.all(MSC == np.array([(1, 0, 0.5j)], dtype = self.dtype)))

    def test_single_sign(self):
        MSC = np.array([(0, 1, 0.5j)], dtype = self.dtype)
        for i in range(5):
            with self.subTest(shift=i):
                shifted = msc.shift(MSC, i, None)
                self.assertEqual(shifted['masks'], 0)
                self.assertEqual(shifted['signs'], 2**i)
                self.assertEqual(shifted['coeffs'], 0.5j)
                self.assertTrue(np.all(MSC == np.array([(0, 1, 0.5j)], dtype = self.dtype)))

    def test_single_mask_wrap(self):
        MSC = np.array([(16, 0, 0.5j)], dtype = self.dtype)
        for i in range(1,5):
            with self.subTest(shift=i):
                shifted = msc.shift(MSC, i, 5)
                self.assertEqual(shifted['masks'], 2**(i-1))
                self.assertEqual(shifted['signs'], 0)
                self.assertEqual(shifted['coeffs'], 0.5j)
                self.assertTrue(np.all(MSC == np.array([(16, 0, 0.5j)], dtype = self.dtype)))

    def test_single_sign_wrap(self):
        MSC = np.array([(0, 16, 0.5j)], dtype = self.dtype)
        for i in range(1,5):
            with self.subTest(shift=i):
                shifted = msc.shift(MSC, i, 5)
                self.assertEqual(shifted['masks'], 0)
                self.assertEqual(shifted['signs'], 2**(i-1))
                self.assertEqual(shifted['coeffs'], 0.5j)
                self.assertTrue(np.all(MSC == np.array([(0, 16, 0.5j)], dtype = self.dtype)))

    def test_multiple(self):
        MSC = np.array([(3, 4, 0.5),
                        (4, 1, 1.5),
                        (1, 3, 4.5j)], dtype = self.dtype)
        orig = MSC.copy()

        shifted = msc.shift(MSC, 2, None)
        self.assertTrue(np.all(shifted['masks'] == MSC['masks']*4))
        self.assertTrue(np.all(shifted['signs'] == MSC['signs']*4))
        self.assertTrue(np.all(shifted['coeffs'] == MSC['coeffs']))
        self.assertTrue(np.all(MSC == orig))

    def test_multiple_wrap(self):
        MSC = np.array([(5, 4, 0.5),
                        (4, 1, 1.5),
                        (1, 3, 4.5j)], dtype = self.dtype)
        orig = MSC.copy()

        shifted = msc.shift(MSC, 3, 5)
        self.assertTrue(np.all(shifted['masks'] == np.array([9, 1, 8])))
        self.assertTrue(np.all(shifted['signs'] == np.array([1, 8, 24])))
        self.assertTrue(np.all(shifted['coeffs'] == MSC['coeffs']))
        self.assertTrue(np.all(MSC == orig))

class ReduceMSC(ut.TestCase):
    '''
    Test the _combine_and_sort method.
    '''

    def setUp(self):
        self.dtype = np.dtype([('masks', np.int32),
                               ('signs', np.int32),
                               ('coeffs', np.complex128)])

    def check_same(self, check, target):
        check = msc.combine_and_sort(check)
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_single(self):
        check = np.array([(5, 2, -0.5j)], dtype=self.dtype)
        target = np.array([(5, 2, -0.5j)], dtype=self.dtype)
        self.check_same(check, target)

    def test_same(self):
        check = np.array([(5, 2, 2), (5, 2, -0.5)], dtype=self.dtype)
        target = np.array([(5, 2, 1.5)], dtype=self.dtype)
        self.check_same(check, target)

    def test_same_zero(self):
        check = np.array([(5, 2, 0.5), (5, 2, -0.5)], dtype=self.dtype)
        target = np.array([], dtype=self.dtype)
        self.check_same(check, target)

    def test_different(self):
        check = np.array([(5, 3, 2), (5, 2, -0.5)], dtype=self.dtype)
        target = np.array([(5, 2, -0.5), (5, 3, 2)], dtype=self.dtype)
        self.check_same(check, target)

    def test_many(self):
        check = np.array([(5, 3, 2),
                          (5, 2, -0.5),
                          (0, 0, 0),
                          (15, 3, 1.5j),
                          (14, 3, 2j),
                          (1, 0, 0),
                          (0, 0, 1.2),
                          (1, 0, 3),
                          (1, 0, -2),
                          (2, 3, 4),
                          (1, 0, -1)],
                         dtype=self.dtype)

        target = np.array([(0, 0, 1.2),
                           (2, 3, 4),
                           (5, 2, -0.5),
                           (5, 3, 2),
                           (14, 3, 2j),
                           (15, 3, 1.5j)],
                          dtype=self.dtype)

        self.check_same(check, target)

class Serialization(ut.TestCase):
    '''
    Test the msc.serialize and msc.deserialize methods.
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
                'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                                 dtype = dtype32.newbyteorder('B')),
                'serial' : b'2\n32\n' + \
                           b'\x00\x00\x00\x01\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x05\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            },
            {
                'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                                 dtype = dtype32.newbyteorder('L')),
                'serial' : b'2\n32\n' + \
                           b'\x00\x00\x00\x01\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x05\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            },
            {
                'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                                 dtype = dtype64.newbyteorder('B')),
                'serial' : b'2\n64\n' + \
                           b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            },
            {
                'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                                 dtype = dtype64.newbyteorder('L')),
                'serial' : b'2\n64\n' + \
                           b'\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            }
        ]

    def test_serialize(self):
        for n, case in enumerate(self.test_cases):
            with self.subTest(n = n):
                ser = msc.serialize(case['MSC'])
                self.assertEqual(ser, case['serial'])

    def test_deserialize(self):
        for n, case in enumerate(self.test_cases):
            with self.subTest(n = n):
                MSC = msc.deserialize(case['serial'])
                self.assertTrue(np.all(MSC == case['MSC']),
                                msg = '\n'+'\n\n'.join([str(MSC), str(case['MSC'])]))

class MaxSpinIdx(ut.TestCase):

    dtype = np.dtype([('masks', np.int32),
                      ('signs', np.int32),
                      ('coeffs', np.complex128)])

    def test_single_zero(self):
        check = np.array([(1, 0, 2)], dtype = self.dtype)
        self.assertEqual(msc.max_spin_idx(check), 0)

    def test_single_mask(self):
        check = np.array([(4, 2, 18j)], dtype = self.dtype)
        self.assertEqual(msc.max_spin_idx(check), 2)

    def test_single_sign(self):
        check = np.array([(1, 3, 18j)], dtype = self.dtype)
        self.assertEqual(msc.max_spin_idx(check), 1)

    def test_multiple_mask(self):
        check = np.array([(1, 3, 18j), (9, 2, 1), (2, 5, 18j)], dtype = self.dtype)
        self.assertEqual(msc.max_spin_idx(check), 3)

    def test_multiple_sign(self):
        check = np.array([(1, 3, 18j), (9, 2, 1), (2, 17, 12)], dtype = self.dtype)
        self.assertEqual(msc.max_spin_idx(check), 4)

    def test_empty(self):
        # we want -1 in this case so that for loops based on this terminate correctly
        check = np.array([], dtype = self.dtype)
        self.assertEqual(msc.max_spin_idx(check), -1)

class NNZ(ut.TestCase):
    '''
    Test the msc.nnz method.
    '''

    dtype = np.dtype([('masks', np.int32),
                      ('signs', np.int32),
                      ('coeffs', np.complex128)])

    def test_empty(self):
        check = np.array([], dtype = self.dtype)
        self.assertEqual(msc.nnz(check), 0)

    def test_single(self):
        check = np.array([(0, 0, 1)], dtype = self.dtype)
        self.assertEqual(msc.nnz(check), 1)

    def test_single_offdiag(self):
        check = np.array([(2, 0, 1)], dtype = self.dtype)
        self.assertEqual(msc.nnz(check), 1)

    def test_multiple_sign(self):
        check = np.array([(2, 3, 1j), (2, 0, 1)], dtype = self.dtype)
        self.assertEqual(msc.nnz(check), 1)

    def test_multiple(self):
        check = np.array([(0, 0, 1),
                          (1, 0, 2),
                          (1, 1, 3),
                          (2, 4, 0.5j)], dtype = self.dtype)
        self.assertEqual(msc.nnz(check), 3)

if __name__ == '__main__':
    ut.main()
