# pylint: disable=W0212, W0104
'''
Unit tests for operators.py.

These tests should NOT require MPI.
'''

import unittest as ut
from unittest.mock import Mock, MagicMock
import numpy as np

from dynamite.operators import Operator, sigmax, sigmay, sigmaz
from dynamite.operators import sigma_plus, sigma_minus, identity, zero
from dynamite.operators import _OperatorStringRep
from dynamite import msc_tools


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

    def test_sigma_plus(self):
        dnm = msc_tools.msc_to_numpy(sigma_plus().msc, (2,2))
        npy = np.array([[0, 2],
                        [0, 0]])
        self.check_same(dnm, npy)

    def test_sigma_minus(self):
        dnm = msc_tools.msc_to_numpy(sigma_minus().msc, (2,2))
        npy = np.array([[0, 0],
                        [2, 0]])
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


class SizeExceptions(ut.TestCase):

    def test_fundamental(self):
        ops = [sigmax, sigmay, sigmaz, sigma_plus, sigma_minus]
        for op in ops:
            with self.subTest(op=op.__name__):
                with self.assertRaises(ValueError):
                    op(63)

                if msc_tools.msc_dtype['masks'].itemsize == 4:
                    with self.assertRaises(ValueError):
                        op(31)


    def test_translations(self):
        ops = [index_sum, index_product]
        for op in ops:
            with self.subTest(op=op.__name__):
                with self.assertRaises(ValueError):
                    op(sigmax(), size=64)

                if msc_tools.msc_dtype['masks'].itemsize == 4:
                    with self.assertRaises(ValueError):
                        op(sigmax(), size=32)


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

    def test_subtract_number(self):
        dnm = sigmax() - 1
        self.check_same_msc(dnm.msc, [(0, 0, -1), (1, 0, 1)])

    def test_rsubtract_number(self):
        dnm = 1 - sigmax()
        self.check_same_msc(dnm.msc, [(0, 0, 1), (1, 0, -1)])

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

    def test_hopping(self):
        ladder = 0.5*(sigma_plus(0)*sigma_minus(1) + sigma_minus(0)*sigma_plus(1))
        pauli  = sigmax(0)*sigmax(1) + sigmay(0)*sigmay(1)
        self.check_same_msc(ladder.msc, pauli.msc)

    def test_divide_op_op(self):
        with self.assertRaises(TypeError):
            sigmax() / sigmay()

    def test_divide_number_op(self):
        with self.assertRaises(TypeError):
            1 / sigmay()

    def test_divide_op_number(self):
        dnm = sigmax() / 2
        self.check_same_msc(dnm.msc, [(1, 0, 0.5)])

    def test_fail_shell(self):
        o1 = sigmax(0)
        o2 = sigmax(1)

        o1.shell = True
        with self.assertRaises(ValueError):
            o1 + o2
        with self.assertRaises(ValueError):
            o1 * o2

        o1.shell = False
        o2.shell = True
        with self.assertRaises(ValueError):
            o1 + o2
        with self.assertRaises(ValueError):
            o1 * o2

    def test_fail_L(self):
        test_vals = [None, 4, 5]

        for L1 in test_vals:
            for L2 in test_vals:
                if L1 is L2:
                    continue

                with self.subTest(L1=L1, L2=L2):
                    o1 = sigmax(0)
                    o2 = sigmax(1)

                    if L1 is not None:
                        o1.L = L1
                    if L2 is not None:
                        o2.L = L2

                    with self.assertRaises(ValueError):
                        o1 + o2
                    with self.assertRaises(ValueError):
                        o1 * o2

    def test_fail_projection(self):
        o1 = sigmax(0)
        o2 = sigmax(1)

        o1.allow_projection = True
        with self.assertRaises(ValueError):
            o1 + o2
        with self.assertRaises(ValueError):
            o1 * o2

        o1.allow_projection = False
        o2.allow_projection = True
        with self.assertRaises(ValueError):
            o1 + o2
        with self.assertRaises(ValueError):
            o1 * o2

    def test_fail_subspaces(self):
        from dynamite.subspaces import Parity

        o1 = sigmax(0)
        o2 = sigmax(1)

        o1.subspace = Parity('even')
        with self.assertRaises(ValueError):
            o1 + o2
        with self.assertRaises(ValueError):
            o1 * o2

    def test_fail_subspaces_2(self):
        from dynamite.subspaces import Parity

        o1 = sigmax(0)
        o2 = sigmax(1)

        o2.subspace = Parity('even')
        with self.assertRaises(ValueError):
            o1 + o2
        with self.assertRaises(ValueError):
            o1 * o2

    def test_fail_subspaces_different(self):
        from dynamite.subspaces import Parity

        o1 = sigmax(0)
        o2 = sigmax(1)

        o1.subspace = Parity('odd')
        o2.subspace = Parity('even')
        with self.assertRaises(ValueError):
            o1 + o2
        with self.assertRaises(ValueError):
            o1 * o2


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


class SumofProducts(ut.TestCase):

    def test_tex(self):
        op = index_sum(index_product(sigmax(), size=3), size=8)
        self.assertEqual(
            op._repr_latex_(),
            r'$\sum\limits_{j=0}^{5}\left[\prod\limits_{i=0}^{2}\sigma^x_{j+i}\right]$'
        )


class WithBrackets(ut.TestCase):
    '''
    Test classmethod _with_brackets
    '''

    def test_string_parens(self):
        class Op:
            string = 'σx + σy'
            brackets = '()'
        out = _OperatorStringRep.with_brackets(Op, 'string')
        self.assertEqual(out, '(σx + σy)')

    def test_string_square(self):
        class Op:
            string = 'σx + σy'
            brackets = '[]'
        out = _OperatorStringRep.with_brackets(Op, 'string')
        self.assertEqual(out, '[σx + σy]')

    def test_string_none(self):
        class Op:
            string = 'σx + σy'
            brackets = ''
        out = _OperatorStringRep.with_brackets(Op, 'string')
        self.assertEqual(out, 'σx + σy')

    def test_tex_parens(self):
        class Op:
            tex = r'\sigma_{x} + \sigma_{y}'
            brackets = '()'
        out = _OperatorStringRep.with_brackets(Op, 'tex')
        self.assertEqual(out, r'\left(\sigma_{x} + \sigma_{y}\right)')

    def test_tex_square(self):
        class Op:
            tex = r'\sigma_{x} + \sigma_{y}'
            brackets = '[]'
        out = _OperatorStringRep.with_brackets(Op, 'tex')
        self.assertEqual(out, r'\left[\sigma_{x} + \sigma_{y}\right]')

    def test_tex_none(self):
        class Op:
            tex = r'\sigma_{x} + \sigma_{y}'
            brackets = ''
        out = _OperatorStringRep.with_brackets(Op, 'tex')
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

    def test_dim_explicit_L(self):
        o = sigmaz(4)
        o.L = 6
        self.assertEqual(o.dim, (64, 64))

    def test_dim_subspace(self):
        from dynamite.subspaces import Full

        subspace = Mock()
        subspace.get_dimension = MagicMock(return_value=7)
        subspace.L = None

        o = sigmaz(4)
        o.L = 6
        o._subspaces.append((subspace, Full()))
        self.assertEqual(o.dim, (7, 64))
        # this should have set the subspace dimension too
        self.assertEqual(subspace.L, 6)

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
        subspace.L = None
        subspace.identical = MagicMock(return_value=False)

        o = sigmaz()
        o.L = 5
        o.add_subspace(Full(), subspace)
        self.assertEqual(o.left_subspace.L, 5)
        self.assertEqual(o.right_subspace.L, 5)

        with self.assertRaises(ValueError):
            o.subspace

    def test_left_subspace(self):

        from dynamite.subspaces import Subspace, Full
        subspace = Mock(spec=Subspace)
        subspace.L = None
        subspace.identical = MagicMock(return_value=False)

        o = sigmaz()
        o.L = 5

        o.add_subspace(subspace, Full())
        self.assertEqual(o.left_subspace.L, 5)
        self.assertEqual(o.right_subspace.L, 5)

        with self.assertRaises(ValueError):
            o.subspace

    def test_subspace(self):

        from dynamite.subspaces import Subspace
        subspace = Mock(spec=Subspace)
        subspace.L = None
        subspace.identical = MagicMock(return_value=False)

        o = sigmaz()
        o.L = 5
        o.add_subspace(subspace)

        self.assertIs(o.left_subspace, subspace)
        self.assertIs(o.right_subspace, subspace)
        self.assertIs(o.subspace, subspace)

        self.assertEqual(o.subspace.L, 5)

    def test_subspace_L_later(self):

        from dynamite.subspaces import Subspace
        subspace = Mock(spec=Subspace)
        subspace.L = None
        subspace.identical = MagicMock(return_value=False)

        o = sigmaz()
        o.add_subspace(subspace)
        o.L = 5

        self.assertEqual(o.subspace.L, 5)

    def test_subspace_L_later_both(self):

        from dynamite.subspaces import Subspace
        left = Mock(spec=Subspace)
        right = Mock(spec=Subspace)
        left.identical = MagicMock(return_value=False)

        left.L = None
        right.L = None

        o = sigmaz()
        o.add_subspace(left, right)
        o.L = 5

        self.assertEqual(o.left_subspace.L, 5)
        self.assertEqual(o.right_subspace.L, 5)

    def test_subspace_set_L(self):

        from dynamite.subspaces import Subspace
        subspace = Mock(spec=Subspace)
        subspace.identical = MagicMock(return_value=False)

        o = sigmaz()
        subspace.L = 5
        o.add_subspace(subspace)

        self.assertEqual(o.L, 5)
        self.assertEqual(o.subspace.L, 5)

    def test_subspace_set_L_left(self):

        from dynamite.subspaces import Subspace
        left = Mock(spec=Subspace)
        right = Mock(spec=Subspace)
        left.identical = MagicMock(return_value=False)

        o = sigmaz()
        left.L = 5
        right.L = None
        o.add_subspace(left, right)

        self.assertEqual(o.L, 5)
        self.assertEqual(left.L, 5)
        self.assertEqual(right.L, 5)

    def test_subspace_set_L_right(self):

        from dynamite.subspaces import Subspace
        left = Mock(spec=Subspace)
        right = Mock(spec=Subspace)
        left.identical = MagicMock(return_value=False)

        o = sigmaz()
        left.L = None
        right.L = 5
        o.add_subspace(left, right)

        self.assertEqual(o.L, 5)
        self.assertEqual(left.L, 5)
        self.assertEqual(right.L, 5)

    def test_subspace_set_L_fail(self):

        from dynamite.subspaces import Subspace
        left = Mock(spec=Subspace)
        right = Mock(spec=Subspace)
        left.identical = MagicMock(return_value=False)

        o = sigmaz()
        left.L = 5
        right.L = 6
        with self.assertRaises(ValueError):
            o.add_subspace(left, right)

    def test_subspace_L_later_fail(self):
        from dynamite.subspaces import Subspace
        left = Mock(spec=Subspace)
        right = Mock(spec=Subspace)
        left.identical = MagicMock(return_value=False)

        left.L = None
        right.L = None

        o = sigmaz()
        o.add_subspace(left, right)

        left.L = 5
        right.L = 6

        with self.assertRaises(ValueError):
            o.L

    def test_subspace_product_state_fail(self):

        from dynamite.subspaces import Subspace, Full
        subspace = Mock(spec=Subspace)
        subspace.L = None
        subspace.identical = MagicMock(return_value=False)
        subspace.product_state_basis = False

        o = sigmaz()
        with self.assertRaises(ValueError):
            o.add_subspace(subspace, Full())

        subspace2 = Mock(spec=Subspace)
        subspace2.product_state_basis = False

        with self.assertRaises(ValueError):
            o.add_subspace(subspace, subspace2)

        # this should be fine
        o.add_subspace(subspace)

    def test_brackets(self):
        o = sigmaz()
        o._string_rep.string = 'str'
        o._string_rep.tex = 'tex'

        o._string_rep.brackets = '()'
        self.assertEqual(o._string_rep.brackets, '()')
        self.assertEqual(o._string_rep.with_brackets('string'), '(str)')
        self.assertEqual(o._string_rep.with_brackets('tex'), r'\left(tex\right)')

        o._string_rep.brackets = '[]'
        self.assertEqual(o._string_rep.brackets, '[]')
        self.assertEqual(o._string_rep.with_brackets('string'), '[str]')
        self.assertEqual(o._string_rep.with_brackets('tex'), r'\left[tex\right]')

        o._string_rep.brackets = ''
        self.assertEqual(o._string_rep.brackets, '')
        self.assertEqual(o._string_rep.with_brackets('string'), 'str')
        self.assertEqual(o._string_rep.with_brackets('tex'), 'tex')

        with self.assertRaises(ValueError):
            o._string_rep.brackets = '<>'

    def test_precompute_diagonal_fail(self):
        # this should only work when shell=True
        o = sigmaz()
        with self.assertRaises(ValueError):
            o.precompute_diagonal

        with self.assertRaises(ValueError):
            o.precompute_diagonal = True


class MSC(ut.TestCase):
    '''
    Test operator MSC setters, getters, etc.
    '''
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

    def test_truncate(self):
        for tol in (None, 1e-10):
            with self.subTest(tol=tol):
                o = Operator()
                o.msc = np.array([(1, 2, 3), (4, 5, 1e-13)], dtype=self.dtype)
                if tol is None:
                    o.truncate()
                else:
                    o.truncate(tol=tol)
                self.assertTrue(np.array_equal(o.msc,
                                               np.array([(1,2,3)], dtype=self.dtype)))

    def test_truncate_adjust_tol(self):
        o = Operator()
        o.msc = np.array([(1, 2, 3), (4, 5, 1e-8)], dtype=self.dtype)
        o.truncate(tol=1e-5)
        self.assertTrue(np.array_equal(o.msc,
                                       np.array([(1,2,3)], dtype=self.dtype)))


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
            op = Operator.from_bytes(t['serial'])
            self.assertTrue(np.array_equal(op.msc, t['MSC']))


class Repr(ut.TestCase):
    '''
    Test that the representation printed in the interactive shell is accurate.
    Ideally it matches the input, in cases where that is possible.
    '''

    def test_paulis(self):
        test_cases = [
            'sigmax(0)',
            'sigmay(0)',
            'sigmaz(0)',
            'sigma_plus(0)',
            'sigma_minus(0)',
            'sigmax(2)',
            'sigmay(2)',
            'sigmaz(2)',
            'sigma_plus(2)',
            'sigma_minus(2)',
            'identity()',
            'zero()'
        ]
        for case in test_cases:
            self.assertEqual(case, repr(eval(case)))

    def test_sums(self):
        test_cases = [
            'sigmax(0) + sigmax(1)',
            'sigmay(0) + sigmax(0)',
            'sigmaz(1) + sigmax(2)',
            'sigma_plus(0) + sigma_minus(0)',
            'index_sum(sigmax(0), size=10)',
            'index_sum(sigmax(0), size=10, boundary="closed")',
            'index_sum(sigmax(0), size=10, start=1)',
        ]
        for case in test_cases:
            self.assertEqual(case, repr(eval(case)))

    def test_products(self):
        test_cases = [
            'sigmax(0)*sigmax(1)',
            'sigmay(0)*sigmax(0)',
            'sigmaz(1)*sigmax(2)',
            'sigma_plus(0)*sigma_minus(0)',
            'index_product(sigmax(0), size=10)',
            'index_product(sigmax(0), size=10, start=1)',
        ]
        for case in test_cases:
            self.assertEqual(case, repr(eval(case)))

    def test_op_sum_product(self):
        test_cases = [
            (op_sum(sigmax(i) for i in range(4)),
             'sigmax(0) + sigmax(1) + sigmax(2) + sigmax(3)'),
            (op_sum((sigmax(i) for i in range(4)), nshow=1),
             'sigmax(0) + sigmax(1) + sigmax(2) + sigmax(3)'),
            (op_product(sigmax(i) for i in range(4)),
             'sigmax(0)*sigmax(1)*sigmax(2)*sigmax(3)'),
            (op_product(sigmax(i)+sigmay(i) for i in range(4)),
             '(sigmax(0) + sigmay(0))*(sigmax(1) + sigmay(1))*(sigmax(2) + sigmay(2))*(sigmax(3) + sigmay(3))')
        ]
        for op, result in test_cases:
            self.assertEqual(repr(op), result)

    def test_misc(self):
        test_cases = [
            'sigmaz(0)*(sigmax(0) + sigmay(1))',
            'index_sum(sigmax(0) + sigmax(1), size=10)'
        ]
        for case in test_cases:
            self.assertEqual(case, repr(eval(case)))

    def test_constants(self):
        test_cases = [
            '1.234*sigmaz(0)',
            '1.234j*sigmaz(0)',
            '1j*sigmaz(0)',
            '(1+1j)*sigmaz(0)',
            '(1.123+1.234j)*sigmaz(0)',
        ]
        for case in test_cases:
            self.assertEqual(case, repr(eval(case)))


class Str(ut.TestCase):
    '''
    Test the result of calling str() on operators.
    '''

    def test_paulis(self):
        test_cases = [
            (sigmax(), 'σx[0]'),
            (sigmay(), 'σy[0]'),
            (sigmaz(), 'σz[0]'),
            (sigma_plus(), 'σ+[0]'),
            (sigma_minus(), 'σ-[0]'),
            (sigmax(2), 'σx[2]'),
            (sigmay(2), 'σy[2]'),
            (sigmaz(2), 'σz[2]'),
            (sigma_plus(2), 'σ+[2]'),
            (sigma_minus(2), 'σ-[2]'),
            (identity(), '1'),
            (zero(), '0'),
        ]
        for op, str_val in test_cases:
            self.assertEqual(str(op), str_val)

    def test_sums(self):
        test_cases = [
            (sigmax(0) + sigmax(1), 'σx[0] + σx[1]'),
            (sigmay(0) + sigmax(0), 'σy[0] + σx[0]'),
            (sigmaz(1) + sigmax(2), 'σz[1] + σx[2]'),
            (sigma_plus(0) + sigma_minus(0), 'σ+[0] + σ-[0]'),
            (
                index_sum(sigmax(0), size=10),
                'index_sum(σx[0], sites 0-9)'
            ),
            (
                index_sum(sigmax(0), size=10, boundary="closed"),
                'index_sum(σx[0], sites 0-9, wrapped)'
            ),
            (
                index_sum(sigmax(0), size=10, start=1),
                'index_sum(σx[0], sites 1-10)'
            ),
        ]
        for op, str_val in test_cases:
            self.assertEqual(str(op), str_val)

    def test_products(self):
        test_cases = [
            (sigmax(0)*sigmax(1), 'σx[0]*σx[1]'),
            (sigmay(0)*sigmax(0), 'σy[0]*σx[0]'),
            (sigmaz(1)*sigmax(2), 'σz[1]*σx[2]'),
            (sigma_plus(0)*sigma_minus(0), 'σ+[0]*σ-[0]'),
            (
                index_product(sigmax(0), size=10),
                'index_product(σx[0], sites 0-9)'
            ),
            (
                index_product(sigmax(0), size=10, start=1),
                'index_product(σx[0], sites 1-10)'
            ),
        ]
        for op, str_val in test_cases:
            self.assertEqual(str(op), str_val)

    def test_op_sum_product(self):
        test_cases = [
            (
                op_sum(sigmax(i) for i in range(4)),
                'σx[0] + σx[1] + σx[2] + ...'
            ),
            (
                op_sum((sigmax(i) for i in range(4)), nshow=1),
                'σx[0] + ...'
            ),
            (
                op_product(sigmax(i) for i in range(4)),
                'σx[0]*σx[1]*σx[2]*σx[3]'
            ),
            (
                op_product(sigmax(i)+sigmay(i) for i in range(4)),
                '(σx[0] + σy[0])*(σx[1] + σy[1])*(σx[2] + σy[2])*(σx[3] + σy[3])'
            )
        ]
        for op, str_val in test_cases:
            self.assertEqual(str(op), str_val)

    def test_misc(self):
        test_cases = [
            (
                sigmaz(0)*(sigmax(0) + sigmay(1)),
                'σz[0]*(σx[0] + σy[1])'
            ),
            (
                index_sum(sigmax(0) + sigmax(1), size=10),
                'index_sum(σx[0] + σx[1], sites 0-8)'
            ),
        ]
        for op, str_val in test_cases:
            self.assertEqual(str(op), str_val)

    def test_constants(self):
        test_cases = [
            (1*sigmaz(0), 'σz[0]'),
            (2*sigmaz(0), '2*σz[0]'),
            (1.234*sigmaz(0), '1.234*σz[0]'),
            (1.23456*sigmaz(0), '1.23456*σz[0]'),
            (1.234j*sigmaz(0), '1.234j*σz[0]'),
            (1j*sigmaz(0), '1j*σz[0]'),
            ((1+1j)*sigmaz(0), '(1+1j)*σz[0]'),
            ((1.123+1.234j)*sigmaz(0), '(1.123+1.234j)*σz[0]')
        ]
        for op, str_val in test_cases:
            with self.subTest(op=op, correct=str_val):
                self.assertEqual(str(op), str_val)


if __name__ == '__main__':
    ut.main()
