'''
Test correctness of matvec for various cases.
'''

import unittest as ut
import numpy as np
import hamiltonians
from dynamite.operators import identity, sigmax, sigmay, index_sum, index_product
from dynamite.subspace import Full, Parity, Auto
from dynamite.states import State

def generate_hamiltonian_tests(cls):
    for H_name in hamiltonians.__all__:
        setattr(cls, 'test_'+H_name, lambda self, n=H_name: self.check_hamiltonian(n))
    return cls

class MPITestCase(ut.TestCase):

    @classmethod
    def check_all_procs(cls, success):

        from dynamite import config
        config.initialize()

        from petsc4py import PETSc

        # workaround since tompi4py() seems to break in some cases w/ GPU
        if PETSc.COMM_WORLD.size > 1:
            CW = PETSc.COMM_WORLD.tompi4py()
            for i in range(CW.size):
                remote = CW.bcast(success, root = i)
                success = success and remote

        return success

    def assertEqual(self, first, second, msg=None):
        success = first == second
        all_success = self.check_all_procs(success)
        if not all_success:
            if success:
                ut.TestCase.assertTrue(self, all_success, msg = 'failed on another process')
            else:
                ut.TestCase.assertEqual(self, first, second, msg)

    def assertLess(self, first, second, msg=None):
        success = first < second
        all_success = self.check_all_procs(success)
        if not all_success:
            if success:
                ut.TestCase.assertTrue(self, all_success, msg = 'failed on another process')
            else:
                ut.TestCase.assertLess(self, first, second, msg)

    def assertTrue(self, expr, msg = None):
        all_success = self.check_all_procs(expr)
        if not all_success:
            if expr:
                ut.TestCase.assertTrue(self, all_success, msg = 'failed on another process')
            else:
                ut.TestCase.assertTrue(self, expr, msg)

class FullSpace(MPITestCase):

    def check_nonzeros(self, state, nonzeros):
        '''
        Check that a vector has the correct nonzeros.

        Parameters
        ----------
        state : dynamite.states.State
            The state containing the vector

        nonzeros : dict
            A dictionary, where the keys are the indices of the nonzero elements
            and the values are the nonzero values
        '''
        # first check that the state's norm is right
        correct_norm = sum(np.abs(v)**2 for v in nonzeros.values())
        self.assertEqual(state.vec.norm(), correct_norm)

        istart, iend = state.vec.getOwnershipRange()
        for idx, val in nonzeros.items():
            if istart <= idx < iend:
                self.assertEqual(state.vec[idx], val, msg = 'idx: %d' % idx)
            else:
                # we have to do this for MPI
                self.assertEqual(0, 0)

    def test_identity(self):
        s = State(state = 3)
        r = identity() * s
        correct = {3 : 1}
        self.check_nonzeros(r, correct)

    def test_spinflip(self):
        H = index_product(sigmax())
        s = State(state = 'U'*H.get_length())
        r = H * s
        correct = {0 : 1}
        self.check_nonzeros(r, correct)

@generate_hamiltonian_tests
class FullHamiltonians(MPITestCase):
    def check_hamiltonian(self, H_name):
        H = getattr(hamiltonians, H_name)()
        bra, ket = H.create_states()

        # ket.set_product(0)
        ket.set_random(seed = 0)
        #ket.vec.set(1)

        H.dot(ket, bra)
        self.assertLess(1E-3, bra.vec.norm(), msg = 'petsc vec norm incorrect')

        ket_np = ket.to_numpy()
        bra_check = bra.to_numpy()

        if ket_np is not None:

            self.assertNotEqual(np.linalg.norm(bra_check), 0, msg = 'numpy vec zero')

            H_np = H.to_numpy()
            bra_np = H_np.dot(ket_np)
            inner_prod = bra_check.dot(bra_np.conj())
            if inner_prod != 0:
                inner_prod /= np.linalg.norm(bra_check) * np.linalg.norm(bra_np)
        else:
            inner_prod = 1

        self.assertLess(np.abs(1 - inner_prod), 1E-9)

@generate_hamiltonian_tests
class Subspaces(MPITestCase):

    def compare_to_full(self, H, x, check_subspace):
        '''
        Compare multiplication under the full Hamiltonian to multiplication
        in the subspace.

        Parameters
        ----------
        H : dynamite.operators.Operator
            The operator to multiply.

        x : dynamite.states.State
            The state to multiply (subspace should be Full)

        check_subspace : dynamite.subspace.Subspace
            The subspace to multiply under.
        '''
        # compare all possible combinations of going to and from the full space
        self.assertTrue(isinstance(x.subspace, Full))

        to_space = identity()
        to_space.left_subspace = check_subspace

        H.left_subspace = Full()
        H.right_subspace = Full()
        correct_full = H * x
        correct_sub = to_space * correct_full

        with self.subTest():
            self.check_f2s(H, x, check_subspace, correct_sub)
        
        with self.subTest():
            self.check_s2f(H, x, check_subspace, correct_sub)
        
        with self.subTest():
            self.check_s2s(H, x, check_subspace, correct_sub)

    def compare_vecs(self, H, correct, check):

        # compare the local portions of the vectors
        istart, iend = correct.vec.getOwnershipRange()
        correct = correct.vec[istart:iend]
        check = check.vec[istart:iend]

        # this is the amount of machine rounding error we can accumulate
        eps = H.nnz * np.finfo(correct.dtype).eps

        diff = np.abs(correct-check)
        max_idx = np.argmax(diff)
        self.assertTrue(np.allclose(correct, check, rtol=0, atol=eps),
                        msg = '\ncorrect: %e\ncheck: %e\nat %d' % (np.abs(correct[max_idx]), np.abs(check[max_idx]), max_idx))

    def check_f2s(self, H, x, check_subspace, correct):
        '''
        check multiplication from full to subspace
        '''
        H.right_subspace = Full()
        H.left_subspace = check_subspace
        result = H * x

        self.compare_vecs(H, correct, result)

    def check_s2f(self, H, x, check_subspace, correct):
        '''
        check multiplication from subspace to full
        '''
        H.right_subspace = check_subspace
        H.left_subspace = Full()

        to_space = identity()
        to_space.left_subspace = check_subspace

        x_sub = to_space * x
        result = H * x_sub
        result_sub = to_space * result

        self.compare_vecs(H, correct, result_sub)

    def check_s2s(self, H, x, check_subspace, correct):
        '''
        check multiplication from subspace to subspace
        '''
        H.right_subspace = check_subspace
        H.left_subspace = check_subspace

        to_space = identity()
        to_space.left_subspace = check_subspace

        x_sub = to_space * x
        result = H * x_sub

        self.compare_vecs(H, correct, result)

    def test_parity_XX_even(self):
        H = index_sum(sigmax(0)*sigmax(1))
        x = State(state = 0)
        sp = Parity('even')
        self.compare_to_full(H, x, sp)

    def test_parity_XX_odd(self):
        H = index_sum(sigmax(0)*sigmax(1))
        x = State(state = 1)
        sp = Parity('odd')
        self.compare_to_full(H, x, sp)

    def test_parity_YY_even(self):
        H = index_sum(sigmay(0)*sigmay(1))
        x = State(state = 0)
        sp = Parity('even')
        self.compare_to_full(H, x, sp)

    def test_parity_YY_odd(self):
        H = index_sum(sigmay(0)*sigmay(1))
        x = State(state = 1)
        sp = Parity('odd')
        self.compare_to_full(H, x, sp)

    def test_multiply_repeat(self):
        '''
        This sequence of events triggered a bug caused by needed data being
        garbage collected. This test ensures that the bug is fixed.
        '''
        for space in [1, 2]:
            with self.subTest(space = space):
                H = hamiltonians.ising()
                sp = Auto(H, (1 << (H.L//2))-space)
                x = State(state = 'random', seed = 0)

                to_space = identity()
                to_space.left_subspace = sp
                x = to_space*x

                from_space = identity()
                from_space.right_subspace = sp
                x = from_space*x
                to_space*x

    def check_hamiltonian(self, H_name):
        for space in [1, 2]:
            with self.subTest(space = space):
                H = getattr(hamiltonians, H_name)()
                sp = Auto(H, (1 << (H.L//2))-space)

                k = State(subspace = sp, state = 'random', seed = 0)
                
                from_space = identity()
                from_space.right_subspace = sp
                ket = from_space*k

                self.compare_to_full(H, ket, sp)

    # TODO: write tests for multiplication from one subspace to a different one

if __name__ == '__main__':
    from dynamite import config
    config.L = 10
    config.shell = False
    #config.initialize(['-start_in_debugger', 'noxterm'])

    ut.main()

    # import cProfile
    # cProfile.run('ut.main()', 'out.prof')
