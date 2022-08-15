'''
Test the reduced density matrix and entropy computation on PETSc vectors.
'''

import numpy as np

import unittest as ut
import dynamite_test_runner as dtr
import unittest as ut

from dynamite import config
from dynamite.subspaces import Parity, Auto, SpinConserve
from dynamite.states import State
from dynamite.computations import reduced_density_matrix, entanglement_entropy, renyi_entropy
from dynamite.tools import complex_enabled

class Explicit(dtr.DynamiteTestCase):
    def setUp(self):
        self.old_L = config.L
        config._L = None

    def tearDown(self):
        config.L = self.old_L

    def compare(self, check, correct):
        eps = 1E-10  # TODO: should compute this from machine epsilon

        if check[0,0] != -1: #  process 0
            valid = np.allclose(check, correct, atol=eps, rtol=0)
            self.assertTrue(valid, msg = '\ncheck: %s\ncorrect: %s' % (str(check), str(correct)))

    def assertClose(self, check, correct):
        self.assertTrue(np.isclose(check, correct),
                        msg='\ncheck: %s\ncorrect: %s' % (str(check), str(correct)))

    def check_entropy(self, state, keep, ent_entropy):
        from dynamite import config
        config._initialize()
        from petsc4py import PETSc

        check = entanglement_entropy(state, keep)

        renyi_check = []
        renyi_powers = [0, 1]
        for alpha in renyi_powers:
            renyi_check.append( renyi_entropy(state, keep, alpha) )

        if PETSc.COMM_WORLD.rank == 0:

            with self.subTest(which='entangle'):
                self.assertClose(check, ent_entropy)

            with self.subTest(which='renyi0'):
                if ent_entropy == 0:
                    renyi0_val = 0
                else:
                    renyi0_val = np.log(2**len(keep))
                self.assertClose(renyi_check[0], renyi0_val)

            with self.subTest(which='renyi1'):
                self.assertClose(renyi_check[1], ent_entropy)

        else:
            for check_val in [check]+renyi_check:
                self.assertEqual(check_val, -1)

    def test_exception_badidx_low(self):
        keep = [-1, 0]
        state = State(L=4, state='U'*4)
        with self.assertRaises(ValueError):
            reduced_density_matrix(state, keep)

    def test_exception_badidx_high(self):
        keep = [0, 1, 50]
        state = State(L=4, state='U'*4)
        with self.assertRaises(ValueError):
            reduced_density_matrix(state, keep)

    def test_exception_wrong_order(self):
        keep = [1, 0]
        state = State(L=4, state='U'*4)
        with self.assertRaises(ValueError):
            reduced_density_matrix(state, keep)

    def test_empty_keep(self):
        state = State(L=2)
        keep = []
        vals = [0.5, 0.5, 0.5, 0.5]
        start, end = state.vec.getOwnershipRange()
        for i in range(start, end):
            state.vec[i] = vals[i]
        state.vec.assemblyBegin()
        state.vec.assemblyEnd()
        state.set_initialized()

        dm = reduced_density_matrix(state, keep)

        correct = np.array(
            [[1]],
            dtype=np.complex128)
        self.compare(dm, correct)

    @ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
    def test_complex_sign(self):
        state = State(L=2)
        keep = [0]
        vals = [1, 1j, 0, 0]
        start, end = state.vec.getOwnershipRange()
        for i in range(start, end):
            state.vec[i] = vals[i]
        state.vec.assemblyBegin()
        state.vec.assemblyEnd()
        state.set_initialized()

        dm = reduced_density_matrix(state, keep)

        correct = np.array(
            [[1, -1j], [1j, 1]],
            dtype=np.complex128)
        self.compare(dm, correct)

    @ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
    def test_L4(self):
        state_vals = [
            (0.03-0.293j),(0.131+0.203j),(0.063+0.17j),(0.027+0.226j),(-0.047+0.292j),
            (0.089+0.183j),(-0.038-0.024j),(0.239+0.171j),(0.283-0.233j),(-0.071+0.085j),
            (0.178-0.218j),(0.018+0.271j),(0.042+0.013j),(0.26-0.052j),(-0.262-0.144j),
            (-0.252+0.167j),
        ]
        state = State(L=4)
        start, end = state.vec.getOwnershipRange()
        for i in range(start, end):
            state.vec[i] = state_vals[i]
        state.vec.assemblyBegin()
        state.vec.assemblyEnd()
        state.set_initialized()

        correct_dms = [
            (
                [0],
                np.array([
                    [ (0.51401+0j),(-0.022913+0.007162j) ],
                    [ (-0.022913-0.007162j),(0.485675+0j) ],
                ]),
                0.6916884573920534,
            ),
            (
                [0,2],
                np.array([
                    [(0.333204+0j),(-0.1112-0.113795j),(-0.099827+0.069346j),(-0.002388-0.022364j)],
                    [(-0.1112+0.113795j),(0.196206+0j),(0.001052-0.11965j),(0.111748-0.009399j) ],
                    [(-0.099827-0.069346j),(0.001052+0.11965j),(0.180806+0j),(0.088287+0.120957j)],
                    [(-0.002388+0.022364j),(0.111748+0.009399j),(0.088287-0.120957j),(0.289469+0j)],
                ]),
                0.9691946314869655,
            ),
            (
                [1,3],
                np.array([
                    [ (0.274002+0j),(0.048837-0.03139j),(0.100159-0.036394j),(0.104984-0.221712j) ],
                    [ (0.048837+0.03139j),(0.173056+0j),(0.046852+0.100822j),(0.017627-0.041444j) ],
                    [ (0.100159+0.036394j),(0.046852-0.100822j),(0.218881+0j),(0.035845+0.013317j)],
                    [ (0.104984+0.221712j),(0.017627+0.041444j),(0.035845-0.013317j),(0.333746+0j)],
                ]),
                0.9691946314869655
            ),
            (
                [2],
                np.array([
                    [ (0.52941+0j),(0.011921+0.059947j) ],
                    [ (0.011921-0.059947j),(0.470275+0j) ],
                ]),
                0.6839923299240713,
            ),
        ]

        for keep, correct, entropy in correct_dms:
            with self.subTest(keep=keep):
                check = reduced_density_matrix(state, keep)
                self.compare(check, correct)
                self.check_entropy(state, keep, entropy)

class Checker(dtr.DynamiteTestCase):
    state = None
    def compare_rdm(self, keep, correct):
        check = reduced_density_matrix(self.state, keep)

        if check[0,0] != -1: # process 0
            eps = 1E-10  # TODO: should compute this from machine epsilon
            correct = np.array(correct, dtype=np.complex128)
            valid = np.allclose(check, correct, atol=eps, rtol=0)
            self.assertTrue(valid, msg = '\ncheck: %s\ncorrect: %s'
                            % (str(check), str(correct)))

class FullSpace(Checker):
    def setUp(self):
        self.state = State()

    def test_product_state(self):
        self.state.set_product(self.state.subspace.idx_to_state(0))
        keep = [0, 1]
        correct = np.zeros((4,4))
        state = self.state.subspace.idx_to_state(0) & 3
        correct[state, state] = 1
        self.compare_rdm(keep, correct)

    def test_trace_first(self):
        from dynamite.operators import identity
        from dynamite.subspaces import Full

        self.state.set_random(seed=0)

        to_full = identity()
        to_full.add_subspace(Full(), self.state.subspace)
        full_state = to_full*self.state

        full_np = full_state.to_numpy()

        for n_trace in range(1, self.state.L-1):
            with self.subTest(n_trace=n_trace):
                keep = list(range(n_trace, self.state.L))

                if len(keep) > self.state.L//2 and 'slow' in self.skip_flags:
                    continue

                if full_np is not None: # process 0
                    full_reshaped = full_np.reshape((full_np.size//(2**n_trace), -1))
                    correct = np.dot(full_reshaped, np.conj(full_reshaped.T))
                else:
                    correct = None

                self.compare_rdm(keep, correct)

    def test_trace_last(self):
        from dynamite.operators import identity
        from dynamite.subspaces import Full

        self.state.set_random(seed=0)

        to_full = identity()
        to_full.add_subspace(Full(), self.state.subspace)
        full_state = to_full*self.state

        full_np = full_state.to_numpy()

        for n_keep in range(1, self.state.L-1):
            with self.subTest(n_keep=n_keep):
                keep = list(range(0, n_keep))

                if len(keep) > self.state.L//2 and 'slow' in self.skip_flags:
                    continue

                if full_np is not None: # process 0
                    full_reshaped = full_np.reshape((full_np.size//(2**n_keep), -1))
                    correct = np.dot(full_reshaped.T, np.conj(full_reshaped))
                else:
                    correct = None

                self.compare_rdm(keep, correct)

class EvenParitySpace(FullSpace):
    def setUp(self):
        self.state = State(subspace=Parity('even'))

class OddParitySpace(FullSpace):
    def setUp(self):
        self.state = State(subspace=Parity('odd'))

class AutoSpace(FullSpace):
    def setUp(self):
        from dynamite.operators import sigmax, sigmay, index_sum
        H = index_sum(sigmax(0)*sigmax(1) + sigmay(0)*sigmay(1))
        self.state = State(subspace=Auto(H, 'U'*(config.L//2) + 'D'*(config.L-config.L//2)))

class SpinConserveSpace(FullSpace):
    def setUp(self):
        self.state = State(subspace=SpinConserve(config.L, config.L//2))

class SpinConserveSpinFlipSpace(FullSpace):
    def test_spinflip_fail(self):
        if config.L % 2:
            self.skipTest("only for even L")

        for sign in '+-':
            with self.subTest(sign=sign):
                state = State(
                    state='random',
                    subspace=SpinConserve(
                        config.L, config.L//2, spinflip=sign
                    )
                )
                with self.assertRaises(ValueError):
                    reduced_density_matrix(state, [0])

if __name__ == '__main__':
    dtr.main()
