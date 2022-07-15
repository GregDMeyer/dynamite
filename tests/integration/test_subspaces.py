'''
Integration tests for states.
'''

import numpy as np

import dynamite_test_runner as dtr

from dynamite import config
from dynamite.states import State, UninitializedError
from dynamite.subspaces import SpinConserve, Auto, Full, Parity

from hamiltonians import localized


class AutoBuild(dtr.DynamiteTestCase):

    def test_full_auto(self):
        for space in [1, 2]:
            for sort in [True, False]:
                with self.subTest(space=space):
                    with self.subTest(sort=sort):
                        H = localized()
                        subspace_dict = {
                            'full': Full(),
                            'auto': Auto(H, (1 << (H.L//2))-space, sort=sort)
                        }
                        for left in ('full', 'auto'):
                            for right in ('full', 'auto'):
                                if left == 'full' and right == 'full':
                                    continue
                                with self.subTest(left=left, right=right):
                                    subspaces = tuple([subspace_dict[x]
                                                       for x in (left, right)])
                                    H.add_subspace(*subspaces)
                                    H.build_mat(subspaces=subspaces)


class ConfigLSetting(dtr.DynamiteTestCase):

    def test_full(self):
        s = Full()
        self.assertEqual(s.L, config.L)

    def test_parity(self):
        for p in ('even', 'odd'):
            with self.subTest(parity=p):
                s = Parity(p)
                self.assertEqual(s.L, config.L)


class SpinFlipConversion(dtr.DynamiteTestCase):

    def setUp(self):
        self.old_L = config.L
        config.L = 4

    def tearDown(self):
        config.L = self.old_L

    def test_conversion_error(self):
        '''
        Check that an error is thrown if the sign of the spinflip subspace
        is not provided.
        '''
        subspace = SpinConserve(L=4, k=2)
        spinflip_state = State(state='UUDD', subspace=subspace)
        with self.assertRaises(ValueError):
            SpinConserve.convert_spinflip(spinflip_state)

    def test_uninitialized_error(self):
        subspace = SpinConserve(L=4, k=2, spinflip='+')
        spinflip_state = State(subspace=subspace)
        with self.assertRaises(UninitializedError):
            SpinConserve.convert_spinflip(spinflip_state)

    def test_explicit_plus_basis(self):
        '''
        Test explicit states for spinflip conversion.
        '''
        subspace = SpinConserve(L=4, k=2, spinflip='+')
        for start_state in [0, 1, 2]:
            with self.subTest(start_state=start_state):
                spinflip_state = State(state=subspace.idx_to_state(start_state)[0], subspace=subspace)
                prod_basis_state = SpinConserve.convert_spinflip(spinflip_state)
                coeffs = [0]*3
                coeffs[start_state] = 1/np.sqrt(2)
                correct = np.array(coeffs + coeffs[::-1])

                prod_basis_state_np = prod_basis_state.to_numpy()
                self.assertTrue(
                    np.all(
                        np.abs(prod_basis_state_np - correct) < 1E-15
                    ) if prod_basis_state_np is not None else True
                )

    def test_explicit_minus_basis(self):
        '''
        Test explicit states for spinflip conversion.
        '''
        subspace = SpinConserve(L=4, k=2, spinflip='-')
        for start_state in [0, 1, 2]:
            with self.subTest(start_state=start_state):
                spinflip_state = State(state=subspace.idx_to_state(start_state)[0], subspace=subspace)
                prod_basis_state = SpinConserve.convert_spinflip(spinflip_state)
                coeffs = [0]*3
                coeffs[start_state] = 1/np.sqrt(2)
                correct = np.array(coeffs + [-x for x in coeffs[::-1]])

                prod_basis_state_np = prod_basis_state.to_numpy()
                self.assertTrue(
                    np.all(
                        np.abs(prod_basis_state_np - correct) < 1E-15
                    ) if prod_basis_state_np is not None else True
                )

    def test_conversion(self):
        '''
        Test conversion of SpinConserve states to the spinflip subspace and
        back again.
        '''
        config.L = self.old_L

        if config.L % 2 != 0:
            self.skipTest("only for even L")

        for sign in ['+', '-']:
            with self.subTest(sign=sign):
                subspace = SpinConserve(config.L, config.L//2, spinflip=sign)
                spinflip_state = State(state='random', subspace=subspace)
                prod_basis_state = SpinConserve.convert_spinflip(spinflip_state)
                spinflip_check_state = SpinConserve.convert_spinflip(prod_basis_state, sign=sign)

                start_state_np = spinflip_state.to_numpy()
                check_state_np = spinflip_check_state.to_numpy()

                if start_state_np is not None:

                    self.assertNotEqual(np.linalg.norm(check_state_np), 0, msg='zero vector output')

                    bad_idxs = np.where(np.abs(check_state_np - start_state_np) > 1E-12)[0]
                    msg = '\n'
                    for idx in bad_idxs:
                        msg += 'at {}: correct: {}  check: {}\n'.format(idx,
                                                                        start_state_np[idx],
                                                                        check_state_np[idx])

                else:
                    bad_idxs = []
                    msg = ''

                self.assertTrue(len(bad_idxs) == 0, msg=msg)


if __name__ == '__main__':
    dtr.main()
