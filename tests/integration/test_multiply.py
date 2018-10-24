'''
Test correctness of matvec for various cases.
'''

import dynamite_test_runner as dtr
import numpy as np
import hamiltonians
from dynamite.msc_tools import msc_dtype
from dynamite.operators import identity, sigmax, sigmay, index_sum, index_product
from dynamite.subspaces import Full, Parity, Auto
from dynamite.states import State

def generate_hamiltonian_tests(cls):
    for H_name in hamiltonians.__all__:
        setattr(cls, 'test_'+H_name, lambda self, n=H_name: self.check_hamiltonian(n))
    return cls

class FullSpace(dtr.DynamiteTestCase):

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
class FullHamiltonians(dtr.DynamiteTestCase):
    def check_hamiltonian(self, H_name):
        H = getattr(hamiltonians, H_name)()
        bra, ket = H.create_states()

        #ket.set_product(0)
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

            bad_idxs = np.where(np.abs(bra_check - bra_np) > 1E-12)[0]
            msg = '\n'
            for idx in bad_idxs:
                msg += 'at {}: correct: {}  check: {}\n'.format(idx, bra_np[idx], bra_check[idx])
        else:
            inner_prod = 1
            msg = ''

        self.assertLess(np.abs(1 - inner_prod), 1E-9, msg=msg)

@generate_hamiltonian_tests
class Subspaces(dtr.DynamiteTestCase):

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
        to_space.add_subspace(check_subspace, Full())

        correct_full = State(subspace=Full())
        correct_sub = State(subspace=check_subspace)
        H.dot(x, correct_full)
        to_space.dot(correct_full, correct_sub)

        with self.subTest(which='f2s'):
            self.check_f2s(H, x, check_subspace, correct_sub)

        with self.subTest(which='s2f'):
            self.check_s2f(H, x, check_subspace, correct_sub)

        with self.subTest(which='s2s'):
            self.check_s2s(H, x, check_subspace, correct_sub)

    def check_f2s(self, H, x, check_subspace, correct):
        '''
        check multiplication from full to subspace
        '''
        H.add_subspace(check_subspace, Full())
        result = State(subspace=check_subspace)
        H.dot(x, result)

        eps = H.nnz*np.finfo(msc_dtype[2]).eps
        self.check_vec_equal(correct, result, eps=eps)

    def check_s2f(self, H, x, check_subspace, correct):
        '''
        check multiplication from subspace to full
        '''
        H.add_subspace(Full(), check_subspace)
        to_space = identity()
        to_space.add_subspace(check_subspace, Full())

        sub_state = State(subspace=check_subspace)
        full_state = State(subspace=Full())

        to_space.dot(x, sub_state)
        H.dot(sub_state, full_state)
        to_space.dot(full_state, sub_state)

        eps = H.nnz*np.finfo(msc_dtype[2]).eps
        self.check_vec_equal(correct, sub_state, eps=eps)

    def check_s2s(self, H, x, check_subspace, correct):
        '''
        check multiplication from subspace to subspace
        '''
        H.add_subspace(check_subspace)
        to_space = identity()
        to_space.add_subspace(check_subspace, Full())

        x_sub = State(subspace=check_subspace)
        result = State(subspace=check_subspace)

        to_space.dot(x, x_sub)
        H.dot(x_sub, result)

        eps = H.nnz*np.finfo(msc_dtype[2]).eps
        self.check_vec_equal(correct, result, eps=eps)

    @classmethod
    def generate_random_in_subspace(cls, space):
        k = State(subspace=space, state='random', seed=0)
        from_space = identity()
        from_space.add_subspace(Full(), space)
        ket = State(subspace=Full())
        from_space.dot(k, ket)
        return ket

    def test_parity_XX_even(self):
        H = index_sum(sigmax(0)*sigmax(1))
        sp = Parity('even')
        x = self.generate_random_in_subspace(sp)
        self.compare_to_full(H, x, sp)

    def test_parity_XX_odd(self):
        H = index_sum(sigmax(0)*sigmax(1))
        sp = Parity('odd')
        x = self.generate_random_in_subspace(sp)
        self.compare_to_full(H, x, sp)

    def test_parity_YY_even(self):
        H = index_sum(sigmay(0)*sigmay(1))
        sp = Parity('even')
        x = self.generate_random_in_subspace(sp)
        self.compare_to_full(H, x, sp)

    def test_parity_YY_odd(self):
        H = index_sum(sigmay(0)*sigmay(1))
        sp = Parity('odd')
        x = self.generate_random_in_subspace(sp)
        self.compare_to_full(H, x, sp)

    def check_hamiltonian(self, H_name):
        for space in [1, 2]:
            for sort in [True, False]:
                with self.subTest(space=space):
                    with self.subTest(sort=sort):
                        H = getattr(hamiltonians, H_name)()
                        sp = Auto(H, (1 << (H.L//2))-space)

                        ket = self.generate_random_in_subspace(sp)

                        self.compare_to_full(H, ket, sp)

    # TODO: write tests for multiplication from one subspace to a different one

if __name__ == '__main__':
    dtr.main()
