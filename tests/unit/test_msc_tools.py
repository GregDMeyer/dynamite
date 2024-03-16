# pylint: disable=W0212
'''
Unit tests for the classmethods of the Operator class in operators.py.

These tests should NOT require MPI.
'''

import unittest as ut
import numpy as np

from dynamite import msc_tools

class ToNumpy(ut.TestCase):
    '''
    Test the msc_tools.MSC_to_numpy method, whose behavior defines the MSC array.
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
        dnm = msc_tools.msc_to_numpy([(0, 0, 1)], (5,5))
        npy = np.identity(5)
        self.check_same(dnm, npy)

    def test_identity_wide(self):
        dnm = msc_tools.msc_to_numpy([(0, 0, 1)], (3,5),
                                     idx_to_state = lambda x: x if x < 3 else -1)
        npy = np.identity(5)[:3,:]
        self.check_same(dnm, npy)

    def test_identity_tall(self):
        def state_to_idx(x):
            rtn = x.copy()
            rtn[rtn >= 3] = -1
            return rtn

        dnm = msc_tools.msc_to_numpy([(0, 0, 1)], (5,3),
                                     state_to_idx = state_to_idx)
        npy = np.identity(5)[:,:3]
        self.check_same(dnm, npy)

    def test_cut_off(self):
        def state_to_idx(x):
            rtn = x.copy()
            rtn[rtn >= 3] = -1
            return rtn

        dnm = msc_tools.msc_to_numpy([(0, 0, 1)], (5,5),
                                     state_to_idx = state_to_idx)
        npy = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.check_same(dnm, npy)

    def test_nonherm_diag(self):
        dnm = msc_tools.msc_to_numpy([(0, 0, 1j)], (2,2))
        npy = np.array(
            [
                [1j, 0],
                [0, 1j],
            ]
        )
        self.check_same(dnm, npy)

    def test_nonherm_offdiag(self):
        dnm = msc_tools.msc_to_numpy([(1, 0, 1j)], (2,2))
        npy = np.array(
            [
                [0, 1j],
                [1j, 0],
            ]
        )
        self.check_same(dnm, npy)

    def test_nonherm_sign(self):
        dnm = msc_tools.msc_to_numpy([(1, 1, 1j)], (2,2))
        npy = np.array(
            [
                [0, -1j],
                [1j, 0],
            ]
        )
        self.check_same(dnm, npy)

    def test_twoterms_tall(self):
        def state_to_idx(x):
            rtn = x.copy()
            rtn[rtn >= 3] = -1
            return rtn

        dnm = msc_tools.msc_to_numpy([(0, 0, 1), (2, 0, 2)], (5,3),
                                     state_to_idx = state_to_idx)
        npy = np.array(
            [
                [1, 0, 2],
                [0, 1, 0],
                [2, 0, 1],
                [0, 2, 0],
                [0, 0, 0],
            ]
        )
        self.check_same(dnm, npy)

    def test_allflip(self):
        dnm = msc_tools.msc_to_numpy([(15, 0, 1)], (16,16))
        npy = np.identity(16)[:,::-1]
        self.check_same(dnm, npy)

    def test_sign1(self):
        dnm = msc_tools.msc_to_numpy([(0, 1, 1)], (16,16))
        npy = np.diag([1, -1]*8)
        self.check_same(dnm, npy)

    def test_sign2(self):
        dnm = msc_tools.msc_to_numpy([(0, 3, 1)], (16,16))
        npy = np.diag([1, -1, -1, 1]*4)
        self.check_same(dnm, npy)

    def test_signL(self):
        dnm = msc_tools.msc_to_numpy([(0, 8, 1)], (16,16))
        npy = np.diag([1]*8 + [-1]*8)
        self.check_same(dnm, npy)

    def test_signL2(self):
        dnm = msc_tools.msc_to_numpy([(0, 9, 1)], (16,16))
        npy = np.diag([1, -1]*4 + [-1, 1]*4)
        self.check_same(dnm, npy)

    def test_full(self):
        dnm = msc_tools.msc_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8))
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

    def test_dense(self):
        dnm = msc_tools.msc_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8), sparse = False)
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
        dnm = msc_tools.msc_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
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
        dnm = msc_tools.msc_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
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
        dnm = msc_tools.msc_to_numpy([(1, 5, 0.5j), (4, 3, -2)], (8, 8),
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

class IsHermitian(ut.TestCase):
    '''
    Test the is_hermitian method.
    '''

    def test_hermitian(self):
        check = np.array([(1, 3, 1j)], dtype=msc_tools.msc_dtype)
        self.assertTrue(msc_tools.is_hermitian(check))

    def test_hermitian_big(self):
        check = np.array([( 3,  0,  1.        +0.j), ( 3,  3, -1.        +0.j),
                          ( 0,  3,  1.        +0.j), ( 6,  0,  1.        +0.j),
                          ( 6,  6, -1.        +0.j), ( 0,  6,  1.        +0.j),
                          (12,  0,  1.        +0.j), (12, 12, -1.        +0.j),
                          ( 0, 12,  1.        +0.j), ( 0,  1,  0.09762701+0.j),
                          ( 0,  2,  0.43037873+0.j), ( 0,  4,  0.20552675+0.j),
                          ( 0,  8,  0.08976637+0.j)], dtype=msc_tools.msc_dtype)
        self.assertTrue(msc_tools.is_hermitian(check))

    def test_nonhermitian(self):
        check = np.array([(1, 3, 3)], dtype=msc_tools.msc_dtype)
        self.assertFalse(msc_tools.is_hermitian(check))

    def test_nonhermitian_big(self):
        check = np.array([( 0,  1,  0.09762701+0.j), ( 0,  2,  0.43037873+0.j),
                          ( 0,  3,  1.        +0.j), ( 0,  4,  0.20552675+0.j),
                          ( 0,  6,  1.        +0.j), ( 0,  8,  0.08976637+0.j),
                          ( 0, 12,  1.        +0.j), ( 3,  0,  1.        +0.j),
                          ( 3,  3, -1.        +0.j), ( 4,  0,  1.        +0.j),
                          ( 4,  4, -1.        +0.j), ( 6,  0,  1.        +0.j),
                          ( 6,  6, -1.        +0.j), (12,  0,  1.        +0.j),
                          (12, 12, -1.        +0.j)], dtype=msc_tools.msc_dtype)
        self.assertFalse(msc_tools.is_hermitian(check))

class MSCSum(ut.TestCase):
    '''
    Test the _MSC_sum method.
    '''

    def setUp(self):
        self.dtype = msc_tools.msc_dtype

    def check_same(self, check, target):
        check = msc_tools.combine_and_sort(check)
        target = msc_tools.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_single(self):
        check = msc_tools.msc_sum(np.array([(1, 2, 3)], dtype=self.dtype))
        target = [(1, 2, 3)]
        self.check_same(check, target)

    def test_2x1(self):
        lst = [
            np.array([(3, 2, 1j), (5, 6, 2)], dtype=self.dtype),
            np.array([(1, 2, 3)], dtype=self.dtype)
        ]
        check = msc_tools.msc_sum(lst)
        target = [(3, 2, 1j), (5, 6, 2), (1, 2, 3)]
        self.check_same(check, target)

    def test_iterable(self):
        check = msc_tools.msc_sum(np.array([(1, 2, i**2)], dtype=self.dtype) for i in range(5))
        target = [(1, 2, 30)]
        self.check_same(check, target)

    def test_empty(self):
        check = msc_tools.msc_sum([])
        target = []
        self.check_same(check, target)

class MSCProduct(ut.TestCase):
    '''
    Test the _MSC_product method.
    '''

    def setUp(self):
        self.dtype = msc_tools.msc_dtype

    def check_same(self, check, target):
        check = msc_tools.combine_and_sort(check)
        target = msc_tools.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_single(self):
        check = msc_tools.msc_product(np.array([(1,2,3)], dtype=self.dtype))
        target = [(1,2,3)]
        self.check_same(check, target)

    def test_XX(self):
        lst = [
            [(1, 0, 2)],
            [(2, 0, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc_tools.msc_product(lst)
        target = [(3, 0, 6)]
        self.check_same(check, target)

    def test_ZZ(self):
        lst = [
            [(0, 1, 2)],
            [(0, 2, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc_tools.msc_product(lst)
        target = [(0, 3, 6)]
        self.check_same(check, target)

    def test_YY(self):
        lst = [
            [(1, 1, 2)],
            [(2, 2, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc_tools.msc_product(lst)
        target = [(3, 3, 6)]
        self.check_same(check, target)

    def test_XZ(self):
        lst = [
            [(1, 0, 2)],
            [(0, 1, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc_tools.msc_product(lst)
        target = [(1, 1, 6)]
        self.check_same(check, target)

    def test_ZX(self):
        lst = [
            [(0, 1, 2)],
            [(1, 0, 3)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc_tools.msc_product(lst)
        target = [(1, 1, -6)]
        self.check_same(check, target)

    def test_1x2x3(self):
        lst = [
            [(1, 0, 1)],
            [(0, 1, 2), (3, 3, 5)],
            [(1, 0, 3), (6, 4, 7), (3, 4, 11)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc_tools.msc_product(lst)
        target = [(0, 1, -6),
                  (7, 5, 14),
                  (2, 5,-22),
                  (3, 3,-15),
                  (4, 7,-35),
                  (1, 7, 55)]
        self.check_same(check, target)

    def test_one_empty(self):
        lst = [
            [],
            [(0, 1, 2), (3, 3, 5)],
            [(1, 0, 3), (6, 4, 7), (3, 4, 11)]
        ]
        lst = [np.array(x, dtype=self.dtype) for x in lst]
        check = msc_tools.msc_product(lst)
        target = []
        self.check_same(check, target)

class ShiftMSC(ut.TestCase):
    '''
    Tests the shift method.
    '''

    def setUp(self):
        self.dtype = msc_tools.msc_dtype

    def test_single_mask(self):
        msc = np.array([(1, 0, 0.5j)], dtype = self.dtype)
        for i in range(5):
            with self.subTest(shift=i):
                shifted = msc_tools.shift(msc, i, None)
                self.assertEqual(shifted['masks'], 2**i)
                self.assertEqual(shifted['signs'], 0)
                self.assertEqual(shifted['coeffs'], 0.5j)
                # check that we haven't changed it
                self.assertTrue(np.all(msc == np.array([(1, 0, 0.5j)], dtype = self.dtype)))

    def test_single_sign(self):
        msc = np.array([(0, 1, 0.5j)], dtype = self.dtype)
        for i in range(5):
            with self.subTest(shift=i):
                shifted = msc_tools.shift(msc, i, None)
                self.assertEqual(shifted['masks'], 0)
                self.assertEqual(shifted['signs'], 2**i)
                self.assertEqual(shifted['coeffs'], 0.5j)
                self.assertTrue(np.all(msc == np.array([(0, 1, 0.5j)], dtype = self.dtype)))

    def test_single_mask_wrap(self):
        msc = np.array([(16, 0, 0.5j)], dtype = self.dtype)
        for i in range(1,5):
            with self.subTest(shift=i):
                shifted = msc_tools.shift(msc, i, 5)
                self.assertEqual(shifted['masks'], 2**(i-1))
                self.assertEqual(shifted['signs'], 0)
                self.assertEqual(shifted['coeffs'], 0.5j)
                self.assertTrue(np.all(msc == np.array([(16, 0, 0.5j)], dtype = self.dtype)))

    def test_single_sign_wrap(self):
        msc = np.array([(0, 16, 0.5j)], dtype = self.dtype)
        for i in range(1,5):
            with self.subTest(shift=i):
                shifted = msc_tools.shift(msc, i, 5)
                self.assertEqual(shifted['masks'], 0)
                self.assertEqual(shifted['signs'], 2**(i-1))
                self.assertEqual(shifted['coeffs'], 0.5j)
                self.assertTrue(np.all(msc == np.array([(0, 16, 0.5j)], dtype = self.dtype)))

    def test_multiple(self):
        msc = np.array([(3, 4, 0.5),
                        (4, 1, 1.5),
                        (1, 3, 4.5j)], dtype = self.dtype)
        orig = msc.copy()

        shifted = msc_tools.shift(msc, 2, None)
        self.assertTrue(np.all(shifted['masks'] == msc['masks']*4))
        self.assertTrue(np.all(shifted['signs'] == msc['signs']*4))
        self.assertTrue(np.all(shifted['coeffs'] == msc['coeffs']))
        self.assertTrue(np.all(msc == orig))

    def test_multiple_wrap(self):
        msc = np.array([(5, 4, 0.5),
                        (4, 1, 1.5),
                        (1, 3, 4.5j)], dtype = self.dtype)
        orig = msc.copy()

        shifted = msc_tools.shift(msc, 3, 5)
        self.assertTrue(np.all(shifted['masks'] == np.array([9, 1, 8])))
        self.assertTrue(np.all(shifted['signs'] == np.array([1, 8, 24])))
        self.assertTrue(np.all(shifted['coeffs'] == msc['coeffs']))
        self.assertTrue(np.all(msc == orig))

class ReduceMSC(ut.TestCase):
    '''
    Test the _combine_and_sort method.
    '''

    def setUp(self):
        self.dtype = msc_tools.msc_dtype

    def check_same(self, check, target):
        check = msc_tools.combine_and_sort(check)
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

class Truncate(ut.TestCase):
    '''
    Test truncation method
    '''
    dtype = msc_tools.msc_dtype

    def test_single(self):
        check =  np.array([(5, 2, -0.5j)], dtype=self.dtype)
        target = np.array([(5, 2, -0.5j)], dtype=self.dtype)
        self.assertTrue(np.array_equal(msc_tools.truncate(check, tol=1e-12), target),
                        msg='\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_zero_tol(self):
        check =  np.array([(1, 2, 0),
                           (2, 3, -1e-13),
                           (5, 2, -0.5j)], dtype=self.dtype)
        target = np.array([(2, 3, -1e-13), (5, 2, -0.5j)], dtype=self.dtype)
        self.assertTrue(np.array_equal(msc_tools.truncate(check, tol=0), target),
                        msg='\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_neg_tol(self):
        check = np.array([(5, 2, -0.5j)], dtype=self.dtype)
        with self.assertRaises(ValueError):
            msc_tools.truncate(check, tol=-1e-12)

    def test_empty(self):
        check = np.array([(5, 2, 0)], dtype=self.dtype)
        target = np.array([], dtype=self.dtype)
        self.assertTrue(np.array_equal(msc_tools.truncate(check, tol=1e-12), target),
                        msg='\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_few_empty(self):
        check = np.array([(5, 2, 0), (4, 1, 0)], dtype=self.dtype)
        target = np.array([], dtype=self.dtype)
        self.assertTrue(np.array_equal(msc_tools.truncate(check, tol=1e-12), target),
                        msg='\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_many(self):
        check = np.array([(0, 0, 1.2),
                          (1, 0, 1.2e-13),
                          (1, 3, 8e-15),
                          (2, 3, 4),
                          (5, 2, -0.5),
                          (5, 3, 2),
                          (6, 2, -1e-14),
                          (6, 4, 1j*1e-15),
                          (14, 3, 2j),
                          (15, 3, 1.5j)],
                         dtype=self.dtype)

        target = np.array([(0, 0, 1.2),
                           (2, 3, 4),
                           (5, 2, -0.5),
                           (5, 3, 2),
                           (14, 3, 2j),
                           (15, 3, 1.5j)],
                          dtype=self.dtype)

        self.assertTrue(np.array_equal(msc_tools.truncate(check, tol=1e-12), target),
                        msg='\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))


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
                ser = msc_tools.serialize(case['MSC'])
                self.assertEqual(ser, case['serial'])

    def test_deserialize(self):
        for n, case in enumerate(self.test_cases):
            with self.subTest(n = n):
                msc = msc_tools.deserialize(case['serial'])
                self.assertTrue(np.all(msc == case['MSC']),
                                msg = '\n'+'\n\n'.join([str(msc), str(case['MSC'])]))

class MaxSpinIdx(ut.TestCase):

    dtype = np.dtype([('masks', np.int32),
                      ('signs', np.int32),
                      ('coeffs', np.complex128)])

    def test_single_zero(self):
        check = np.array([(1, 0, 2)], dtype = self.dtype)
        self.assertEqual(msc_tools.max_spin_idx(check), 0)

    def test_single_mask(self):
        check = np.array([(4, 2, 18j)], dtype = self.dtype)
        self.assertEqual(msc_tools.max_spin_idx(check), 2)

    def test_single_sign(self):
        check = np.array([(1, 3, 18j)], dtype = self.dtype)
        self.assertEqual(msc_tools.max_spin_idx(check), 1)

    def test_multiple_mask(self):
        check = np.array([(1, 3, 18j), (9, 2, 1), (2, 5, 18j)], dtype = self.dtype)
        self.assertEqual(msc_tools.max_spin_idx(check), 3)

    def test_multiple_sign(self):
        check = np.array([(1, 3, 18j), (9, 2, 1), (2, 17, 12)], dtype = self.dtype)
        self.assertEqual(msc_tools.max_spin_idx(check), 4)

    def test_empty(self):
        # we want -1 in this case so that for loops based on this terminate correctly
        check = np.array([], dtype = self.dtype)
        self.assertEqual(msc_tools.max_spin_idx(check), -1)

class NNZ(ut.TestCase):
    '''
    Test the msc_tools.nnz method.
    '''

    dtype = np.dtype([('masks', np.int32),
                      ('signs', np.int32),
                      ('coeffs', np.complex128)])

    def test_empty(self):
        check = np.array([], dtype = self.dtype)
        self.assertEqual(msc_tools.nnz(check), 0)

    def test_single(self):
        check = np.array([(0, 0, 1)], dtype = self.dtype)
        self.assertEqual(msc_tools.nnz(check), 1)

    def test_single_offdiag(self):
        check = np.array([(2, 0, 1)], dtype = self.dtype)
        self.assertEqual(msc_tools.nnz(check), 1)

    def test_multiple_sign(self):
        check = np.array([(2, 3, 1j), (2, 0, 1)], dtype = self.dtype)
        self.assertEqual(msc_tools.nnz(check), 1)

    def test_multiple(self):
        check = np.array([(0, 0, 1),
                          (1, 0, 2),
                          (1, 1, 3),
                          (2, 4, 0.5j)], dtype = self.dtype)
        self.assertEqual(msc_tools.nnz(check), 3)

class Table(ut.TestCase):

    def test_empty_L5(self):
        L = 5
        msc = []
        correct = '  coeff. | operator \n' +\
                  '====================\n'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_identity(self):
        L = 5
        msc = [(0, 0, 2.3)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   2.300 | -----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_identity_L10(self):
        L = 10
        msc = [(0, 0, 2.3)]
        correct = '  coeff. |  operator  \n' +\
                  '======================\n' +\
                  '   2.300 | ----------'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_sigmax_0(self):
        L = 5
        msc = [(1, 0, 1)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   1.000 | X----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_sigmaz_0(self):
        L = 5
        L = 5
        msc = [(0, 1, 1)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   1.000 | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_sigmay_0(self):
        L = 5
        msc = [(1, 1, 1j)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   1.000 | Y----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_sigmay_2(self):
        L = 5
        msc = [(4, 4, 1j)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   1.000 | --Y--'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_sigmax_0_coeff(self):
        L = 5
        msc = [(1, 0, 3.141592)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   3.142 | X----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_XX(self):
        L = 5
        msc = [(3, 0, 1)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   1.000 | XX---'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_ZZ(self):
        L = 5
        msc = [(0, 3, 1)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   1.000 | ZZ---'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_XYZ(self):
        L = 5
        msc = [(3, 6, 0.5j)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   0.500 | XYZ--'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_three(self):
        L = 5
        msc = [(1, 0, 2.3), (3, 1, 2j), (3, 6, 0.5j)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   2.300 | X----\n' +\
                  '   2.000 | YX---\n' +\
                  '   0.500 | XYZ--'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_ZZ_wrap(self):
        L = 5
        msc = [(0, 3, 0.25), (0, 6, 0.25), (0, 12, 0.25), (0, 24, 0.25), (0, 17, 0.25),]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   0.250 | ZZ---\n' +\
                  '   0.250 | -ZZ--\n' +\
                  '   0.250 | --ZZ-\n' +\
                  '   0.250 | ---ZZ\n' +\
                  '   0.250 | Z---Z'

        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_large_coeff(self):
        L = 5
        msc = [(0, 1, 1E+9)]
        correct = '  coeff.  | operator \n' +\
                  '=====================\n' +\
                  ' 1.00e+09 | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_small_coeff(self):
        L = 5
        msc = [(0, 1, 1E-9)]
        correct = '  coeff.  | operator \n' +\
                  '=====================\n' +\
                  ' 1.00e-09 | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_long_coeff(self):
        L = 5
        msc = [(0, 1, 1.23456789)]
        # we want it to round
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   1.235 | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_zero_coeff(self):
        L = 5
        msc = [(0, 1, 0)]
        # we want it to round
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '   0.000 | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_imag_int(self):
        L = 5
        msc = [(0, 1, 1j)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '  1.000j | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_mixed_int(self):
        L = 5
        msc = [(0, 1, 1+1j)]
        correct = '   coeff.   | operator \n' +\
                  '=======================\n' +\
                  ' 1.00+1.00j | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_imag_float(self):
        L = 5
        msc = [(0, 1, 1.2501j)]
        correct = '  coeff. | operator \n' +\
                  '====================\n' +\
                  '  1.250j | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))

    def test_mixed_float(self):
        L = 5
        msc = [(0, 1, 1.231+1.341j)]
        correct = '   coeff.   | operator \n' +\
                  '=======================\n' +\
                  ' 1.23+1.34j | Z----'
        check = msc_tools.table(msc, L)
        self.assertEqual(check, correct, msg = '\n' + '\n\n'.join([check, correct]))


if __name__ == '__main__':
    ut.main()
