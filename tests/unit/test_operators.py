# pylint: disable=W0212, W0104
'''
Unit tests for operators.py.

These tests should NOT require MPI.
'''

import unittest as ut
from unittest.mock import Mock, MagicMock
import numpy as np

from dynamite.operators import Operator, sigmax, sigmay, sigmaz, identity, zero
from dynamite import msc_tools

# TODO: add test: get_length on identity should fail without explicit L

class Fundamental(ut.TestCase):

    def check_same(self, dnm, npy):
        '''
        Helper function to check that dynamite and numpy arrays are equal, and
        print the differences if not.
        '''
        self.assertTrue(np.all(dnm == npy),
                        msg = '\n\n'.join(['\ndnm:\n'+str(dnm), '\nnpy:\n'+str(npy)]))

    def test_sigmax(self):
        dnm = msc_tools.msc_to_numpy(sigmax().msc, (2,2))
        npy = np.array([[0, 1],
                        [1, 0]])
        self.check_same(dnm, npy)

    def test_sigmay(self):
        dnm = msc_tools.msc_to_numpy(sigmay().msc, (2,2))
        npy = np.array([[0, -1j],
                        [1j,  0]])
        self.check_same(dnm, npy)

    def test_sigmaz(self):
        dnm = msc_tools.msc_to_numpy(sigmaz().msc, (2,2))
        npy = np.array([[1,  0],
                        [0, -1]])
        self.check_same(dnm, npy)

    def test_identity(self):
        dnm = msc_tools.msc_to_numpy(identity().msc, (2,2))
        npy = np.array([[1, 0],
                        [0, 1]])
        self.check_same(dnm, npy)

    def test_zero(self):
        dnm = msc_tools.msc_to_numpy(zero().msc, (2,2))
        npy = np.array([[0, 0],
                        [0, 0]])
        self.check_same(dnm, npy)

    def test_antihermitian(self):
        dtype = np.dtype([('masks', np.int32),
                          ('signs', np.int32),
                          ('coeffs', np.complex128)])
        dnm = msc_tools.msc_to_numpy(np.array([(1, 1, 1)], dtype=dtype), (2,2))
        # -1j * sigmay
        npy = np.array([[0, -1],
                        [1,  0]])
        self.check_same(dnm, npy)

class UnaryBinary(ut.TestCase):
    '''
    Check the unary and binary operator methods.
    '''

    dtype = msc_tools.msc_dtype

    def check_same_msc(self, check, target):
        check = msc_tools.combine_and_sort(check)
        target = msc_tools.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_sum_same(self):
        dnm = sigmay() + sigmay()
        self.check_same_msc(dnm.msc, [(1, 1, 2j)])

    def test_sum_diff_op(self):
        dnm = sigmax() + sigmay()
        self.check_same_msc(dnm.msc, [(1, 0, 1), (1, 1, 1j)])

    def test_sum_diff_site(self):
        dnm = sigmay(0) + sigmay(1)
        self.check_same_msc(dnm.msc, [(1, 1, 1j), (2, 2, 1j)])

    def test_sum_number(self):
        dnm = 2 + sigmax()
        self.check_same_msc(dnm.msc, [(0, 0, 2), (1, 0, 1)])

    def test_rsum_number(self):
        dnm = sigmax() + 2
        self.check_same_msc(dnm.msc, [(0, 0, 2), (1, 0, 1)])

    def test_product_same(self):
        dnm = sigmay() * sigmay()
        self.check_same_msc(dnm.msc, [(0, 0, 1)])

    def test_product_diff_op(self):
        dnm = sigmax() * sigmay()
        self.check_same_msc(dnm.msc, [(0, 1, 1j)])

    def test_product_diff_site(self):
        dnm = sigmay(0) * sigmay(1)
        self.check_same_msc(dnm.msc, [(3, 3, -1)])

    def test_product_1x2(self):
        dnm = sigmay(0) * (sigmax(1) + sigmay(1))
        self.check_same_msc(dnm.msc, [(3, 1, 1j), (3, 3, -1)])

    def test_product_2x2(self):
        dnm = (2*sigmax(0) + 3*sigmaz(0)) * (5*sigmax(1) + 7*sigmay(1))
        self.check_same_msc(dnm.msc, [(3, 0, 10),
                                      (3, 2, 14j),
                                      (2, 1, 15),
                                      (2, 3, 21j)])

    def test_product_1x2x2(self):
        dnm = sigmax(2) * (2*sigmax(0) + 3*sigmaz(0)) * (5*sigmax(1) + 7*sigmay(1))
        self.check_same_msc(dnm.msc, [(7, 0, 10),
                                      (7, 2, 14j),
                                      (6, 1, 15),
                                      (6, 3, 21j)])

    def test_product_num(self):
        dnm = 4 * sigmay()
        self.check_same_msc(dnm.msc, [(1, 1, 4j)])

    def test_rproduct_num(self):
        dnm = sigmay() * 4
        self.check_same_msc(dnm.msc, [(1, 1, 4j)])

from dynamite.operators import op_sum, op_product

class Sums(ut.TestCase):

    dtype = msc_tools.msc_dtype

    def check_same(self, check, target):
        check = msc_tools.combine_and_sort(check)
        target = msc_tools.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_empty(self):
        dnm = op_sum([])
        self.check_same(dnm.msc, [])

    def test_single(self):
        dnm = op_sum([sigmay()])
        self.check_same(dnm.msc, [(1, 1, 1j)])

    def test_same(self):
        dnm = op_sum([sigmay(), sigmay()])
        self.check_same(dnm.msc, [(1, 1, 2j)])

    def test_diff_op(self):
        dnm = op_sum([sigmax(), sigmay()])
        self.check_same(dnm.msc, [(1, 0, 1), (1, 1, 1j)])

    def test_diff_site(self):
        dnm = op_sum([sigmay(0), sigmay(1)])
        self.check_same(dnm.msc, [(1, 1, 1j), (2, 2, 1j)])

    def test_large_generator(self):
        dnm = op_sum((i+1)*sigmax(i) for i in range(0,15,2))
        self.check_same(dnm.msc, [(1, 0, 1),
                                  (4, 0, 3),
                                  (16, 0, 5),
                                  (64, 0, 7),
                                  (256, 0, 9),
                                  (1024, 0, 11),
                                  (4096, 0, 13),
                                  (16384, 0, 15)])

    def test_large_same(self):
        dnm = op_sum(sigmaz(0)*sigmaz(k) for k in range(1,5))
        self.check_same(dnm.msc, [(0, 3, 1),
                                  (0, 5, 1),
                                  (0, 9, 1),
                                  (0, 17, 1)])

    def test_index_sum(self):
        dnm = op_sum([sigmaz(i)*sigmaz(i+1) for i in range(4)])
        self.check_same(dnm.msc, [(0, 3, 1),
                                  (0, 6, 1),
                                  (0, 12, 1),
                                  (0, 24, 1)])

    def test_large_repeat_product(self):
        ops = [op_sum([sigmaz(i)*sigmaz(i+k) for i in range(5-k)]) for k in range(1,5)]
        dnm = op_sum(op*op for op in ops)
        self.check_same(dnm.msc, [(0, 0, 10),
                                  (0, 5,  2),
                                  (0, 10, 2),
                                  (0, 20, 2),
                                  (0, 17, 2),
                                  (0, 30, 4),
                                  (0, 15, 4),
                                  (0, 27, 4)])

class Products(ut.TestCase):

    dtype = msc_tools.msc_dtype

    def check_same(self, check, target):
        check = msc_tools.combine_and_sort(check)
        target = msc_tools.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_empty(self):
        dnm = op_product([])
        self.check_same(dnm.msc, [(0, 0, 1)])

    def test_single(self):
        dnm = op_product([sigmay()])
        self.check_same(dnm.msc, [(1, 1, 1j)])

    def test_same(self):
        dnm = op_product([sigmay(), sigmay()])
        self.check_same(dnm.msc, [(0, 0, 1)])

    def test_diff_op(self):
        dnm = op_product([sigmax(), sigmay()])
        self.check_same(dnm.msc, [(0, 1, 1j)])

    def test_diff_site(self):
        dnm = op_product([sigmay(0), sigmay(1)])
        self.check_same(dnm.msc, [(3, 3, -1)])

from dynamite.operators import index_sum, index_product

class IndexSum(ut.TestCase):
    '''
    Tests for the index_sum function.
    '''

    dtype = msc_tools.msc_dtype

    def check_same(self, check, target):
        check = msc_tools.combine_and_sort(check)
        target = msc_tools.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_zero(self):
        dnm = index_sum(zero(), size = 5)
        self.check_same(dnm.msc, [])

    def test_one(self):
        dnm = index_sum(sigmax(), size = 1)
        self.check_same(dnm.msc, [(1, 0, 1)])

    def test_one_shifted(self):
        dnm = index_sum(sigmax(), size = 1, start = 2)
        self.check_same(dnm.msc, [(4, 0, 1)])

    def test_single(self):
        dnm = index_sum(sigmax(), size = 5)
        self.check_same(dnm.msc, [(1, 0, 1),
                                  (2, 0, 1),
                                  (4, 0, 1),
                                  (8, 0, 1),
                                  (16, 0, 1)])

    def test_single_start1(self):
        dnm = index_sum(sigmax(), size = 5, start = 1)
        self.check_same(dnm.msc, [(2, 0, 1),
                                  (4, 0, 1),
                                  (8, 0, 1),
                                  (16, 0, 1),
                                  (32, 0, 1)])

    def test_single_wrap(self):
        dnm = index_sum(sigmax(), size = 5, boundary = 'closed')
        self.check_same(dnm.msc, [(1, 0, 1),
                                  (2, 0, 1),
                                  (4, 0, 1),
                                  (8, 0, 1),
                                  (16, 0, 1)])

    def test_twosite(self):
        dnm = index_sum(sigmaz(0)*sigmaz(1), size = 5)
        self.check_same(dnm.msc, [(0, 3, 1),
                                  (0, 6, 1),
                                  (0, 12, 1),
                                  (0, 24, 1)])

    def test_twosite_start1(self):
        dnm = index_sum(sigmaz(0)*sigmaz(1), size = 5, start = 1)
        self.check_same(dnm.msc, [(0, 6, 1),
                                  (0, 12, 1),
                                  (0, 24, 1),
                                  (0, 48, 1)])

    def test_twosite_wrap(self):
        dnm = index_sum(sigmaz(0)*sigmaz(1), size = 5, boundary = 'closed')
        self.check_same(dnm.msc, [(0, 3, 1),
                                  (0, 6, 1),
                                  (0, 12, 1),
                                  (0, 24, 1),
                                  (0, 17, 1)])

    def test_autosize_L(self):
        o = sigmaz(0)*sigmaz(1)
        o.L = 5
        dnm = index_sum(o)
        self.check_same(dnm.msc, [(0, 3, 1),
                                  (0, 6, 1),
                                  (0, 12, 1),
                                  (0, 24, 1)])

    def test_toobig(self):
        with self.assertRaises(ValueError):
            index_sum(sigmaz(0) + sigmaz(4), size = 3)

    def test_size_0(self):
        with self.assertRaises(ValueError):
            index_sum(sigmaz(0), size = 0)

class IndexProduct(ut.TestCase):
    '''
    Tests for the index_product function.
    '''

    dtype = msc_tools.msc_dtype

    def check_same(self, check, target):
        check = msc_tools.combine_and_sort(check)
        target = msc_tools.combine_and_sort(np.array(target, dtype=self.dtype))
        self.assertTrue(np.array_equal(check, target),
                        msg = '\ncheck:\n'+str(check) + '\ntarget:\n'+str(target))

    def test_zero(self):
        dnm = index_product(zero(), size = 5)
        self.check_same(dnm.msc, [])

    def test_one(self):
        dnm = index_product(sigmax(), size = 1)
        self.check_same(dnm.msc, [(1, 0, 1)])

    def test_one_shifted(self):
        dnm = index_product(sigmax(), size = 1, start = 2)
        self.check_same(dnm.msc, [(4, 0, 1)])

    def test_single(self):
        dnm = index_product(sigmax(), size = 5)
        self.check_same(dnm.msc, [(31, 0, 1)])

    def test_single_start1(self):
        dnm = index_product(sigmax(), size = 5, start = 1)
        self.check_same(dnm.msc, [(62, 0, 1)])

    def test_twosite(self):
        dnm = index_product(sigmaz(0) + sigmaz(1), size = 3)
        self.check_same(dnm.msc, [(0, 3, 1),
                                  (0, 5, 1),
                                  (0, 0, 1),
                                  (0, 6, 1)])

    def test_twosite_start1(self):
        dnm = index_product(sigmaz(0) + sigmaz(1), size = 3, start = 1)
        self.check_same(dnm.msc, [(0, 6, 1),
                                  (0, 10, 1),
                                  (0, 0, 1),
                                  (0, 12, 1)])

    def test_autosize_L(self):
        o = sigmaz(0) + sigmaz(1)
        o.L = 3
        dnm = index_product(o)
        self.check_same(dnm.msc, [(0, 3, 1),
                                  (0, 5, 1),
                                  (0, 0, 1),
                                  (0, 6, 1)])

    def test_toobig(self):
        with self.assertRaises(ValueError):
            index_product(sigmaz(0) + sigmaz(4), size = 3)

    def test_size_0(self):
        self.assertEqual(index_product(sigmaz(0), size=0), identity())

class WithBrackets(ut.TestCase):
    '''
    Test classmethod _with_brackets
    '''

    def test_string_parens(self):
        out = Operator._with_brackets('σx + σy', '()', False)
        self.assertEqual(out, '(σx + σy)')

    def test_string_square(self):
        out = Operator._with_brackets('σx + σy', '[]', False)
        self.assertEqual(out, '[σx + σy]')

    def test_string_none(self):
        out = Operator._with_brackets('σx + σy', '', False)
        self.assertEqual(out, 'σx + σy')

    def test_tex_parens(self):
        out = Operator._with_brackets(r'\sigma_{x} + \sigma_{y}', '()', True)
        self.assertEqual(out, r'\left(\sigma_{x} + \sigma_{y}\right)')

    def test_tex_square(self):
        out = Operator._with_brackets(r'\sigma_{x} + \sigma_{y}', '[]', True)
        self.assertEqual(out, r'\left[\sigma_{x} + \sigma_{y}\right]')

    def test_tex_none(self):
        out = Operator._with_brackets(r'\sigma_{x} + \sigma_{y}', '', True)
        self.assertEqual(out, r'\sigma_{x} + \sigma_{y}')

class Properties(ut.TestCase):

    def test_max_spin_idx(self):
        o = sigmaz(4)
        self.assertEqual(o.max_spin_idx, 4)

        # check that it works again, since we save it
        self.assertEqual(o.max_spin_idx, 4)

        # make sure this changes when we change MSC
        o.msc = np.array([(2, 0, 1)], dtype = o.msc.dtype)
        self.assertEqual(o.max_spin_idx, 1)

    def test_L(self):
        o = sigmaz(4)
        self.assertEqual(o.get_length(), 5)
        self.assertIs(o.L, None)

        o.L = 7
        self.assertEqual(o.get_length(), 7)
        self.assertIs(o.L, 7)

        # too small, operator would be off the end of chain
        with self.assertRaises(ValueError):
            o.L = 4

    def test_dim(self):
        o = sigmaz(4)
        self.assertEqual(o.dim, (32, 32))

        o.L = 6
        self.assertEqual(o.dim, (64, 64))

        from dynamite.subspaces import Full

        subspace = Mock()
        subspace.get_dimension = MagicMock(return_value = 7)
        o._subspaces.append((subspace, Full()))
        self.assertEqual(o.dim, (7, 64))

        o.L = 5
        self.assertEqual(o.dim, (7, 32))
        # this should have set the subspace dimension too
        self.assertEqual(o.left_subspace.L, 5)

        o._subspaces[1] = (subspace, subspace)
        self.assertEqual(o.dim, (7, 7))

    def test_nnz(self):
        o = sigmaz(0) + sigmaz(1) + sigmax(0) + sigmax(1)
        self.assertEqual(o.nnz, 3)

    def test_density(self):
        o = sigmaz(0) + sigmaz(1) + sigmax(0) + sigmax(1)
        o.L = 4
        self.assertEqual(o.density, 0.1875)

        o.right_subspace.get_dimension = MagicMock(return_value = 8)

        self.assertEqual(o.density, 0.375)

    def test_shell(self):
        o = Operator()
        self.assertEqual(o.shell, False)

        o.shell = True
        self.assertTrue(o.shell)

        with self.assertRaises(ValueError):
            o.shell = 'gpu'

        with self.assertRaises(ValueError):
            o.shell = 'crab'

    def test_right_subspace(self):

        from dynamite.subspaces import Subspace, Full
        subspace = Mock(spec=Subspace)

        o = sigmaz()
        o.L = 5
        o.add_subspace(Full(), subspace)
        self.assertEqual(o.right_subspace.L, 5)
        o.L = 6
        self.assertEqual(o.right_subspace.L, 6)

        with self.assertRaises(ValueError):
            o.subspace

    def test_left_subspace(self):

        from dynamite.subspaces import Subspace, Full
        subspace = Mock(spec=Subspace)

        o = sigmaz()
        o.L = 5
        o.add_subspace(subspace, Full())
        self.assertEqual(o.left_subspace.L, 5)
        o.L = 6
        self.assertEqual(o.left_subspace.L, 6)

        with self.assertRaises(ValueError):
            o.subspace

    def test_subspace(self):

        from dynamite.subspaces import Subspace
        subspace = Mock(spec=Subspace)

        o = sigmaz()
        o.L = 5
        o.add_subspace(subspace)

        self.assertIs(o.left_subspace, subspace)
        self.assertIs(o.right_subspace, subspace)
        self.assertIs(o.subspace, subspace)

        self.assertEqual(o.subspace.L, 5)
        o.L = 6
        self.assertEqual(o.subspace.L, 6)

    def test_brackets(self):
        o = sigmaz()
        o.string = 'str'
        o.tex = 'tex'

        o.brackets = '()'
        self.assertEqual(o.brackets, '()')
        self.assertEqual(o.with_brackets(which = 'string'), '(str)')
        self.assertEqual(o.with_brackets(which = 'tex'), r'\left(tex\right)')

        o.brackets = '[]'
        self.assertEqual(o.brackets, '[]')
        self.assertEqual(o.with_brackets(which = 'string'), '[str]')
        self.assertEqual(o.with_brackets(which = 'tex'), r'\left[tex\right]')

        o.brackets = ''
        self.assertEqual(o.brackets, '')
        self.assertEqual(o.with_brackets(which = 'string'), 'str')
        self.assertEqual(o.with_brackets(which = 'tex'), 'tex')

        with self.assertRaises(ValueError):
            o.brackets = '<>'

class MSC(ut.TestCase):
    '''
    Test operator MSC setters, getters, etc.
    '''
    # TODO: need to isolate dtype from backend for testing
    # probably can mock it
    dtype = msc_tools.msc_dtype

    def test_set_list(self):
        o = Operator()
        o.msc = [(1, 2, 3)]
        self.assertTrue(isinstance(o.msc, np.ndarray))
        self.assertTrue(o.msc.dtype == self.dtype)
        self.assertTrue(np.all(o.msc == np.array([(1, 2, 3)], dtype=self.dtype)))

    def test_set_array(self):
        o = Operator()
        o.msc = np.array([(1, 2, 3)], dtype=self.dtype)
        self.assertTrue(isinstance(o.msc, np.ndarray))
        self.assertTrue(o.msc.dtype == self.dtype)
        self.assertTrue(np.all(o.msc == np.array([(1, 2, 3)], dtype=self.dtype)))

    def test_reduce(self):
        o = Operator()

        o.msc = [(1, 2, 3), (1, 2, 6)]
        self.assertFalse(o.is_reduced)
        o.reduce_msc()
        self.assertTrue(o.is_reduced)
        self.assertTrue(np.all(o.msc == np.array([(1, 2, 9)], dtype=self.dtype)))

        o.msc = [(1, 2, 3), (4, 5, 6)]
        self.assertFalse(o.is_reduced)
        o.is_reduced = True
        self.assertTrue(o.is_reduced)

from dynamite.operators import from_bytes
class FromBytes(ut.TestCase):

    def test_simple(self):
        test_cases = [{
            'MSC' : np.array([(1, 5, -0.2j), (0, 1, 2)],
                             dtype = msc_tools.msc_dtype),
            'serial' : b'2\n32\n' + \
                           b'\x00\x00\x00\x01\x00\x00\x00\x00' + \
                           b'\x00\x00\x00\x05\x00\x00\x00\x01' + \
                           b'\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xc9\x99\x99\x99\x99\x99\x9a' + \
                           b'@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
        }]

        for t in test_cases:
            op = from_bytes(t['serial'])
            self.assertTrue(np.array_equal(op.msc, t['MSC']))

if __name__ == '__main__':
    ut.main()
