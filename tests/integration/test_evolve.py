
import numpy as np
from scipy.sparse import linalg
import hamiltonians

import dynamite_test_runner as dtr
import unittest as ut

from dynamite import config
from dynamite.operators import sigmax, index_product, identity
from dynamite.states import State
from dynamite.subspaces import Parity
from dynamite.tools import complex_enabled

@ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
class Analytic(dtr.DynamiteTestCase):
    '''
    Some cases in which we can easily write down the answer.
    '''

    def test_pipulse(self):
        H = index_product(sigmax())
        bra, ket = H.create_states()
        bra_check = bra.copy()

        ket.set_product('D'*config.L)
        bra_check.set_product('U'*config.L)

        H.evolve(ket, t = np.pi/2, result = bra)
        self.assertLess(np.abs(1 - np.abs(bra.dot(bra_check))), 1E-9)

@ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
class EvolveChecker(dtr.DynamiteTestCase):
    def evolve_check(self, H, t):
        bra, ket = H.create_states()
        ket.set_random(seed = 0)

        H_np = H.to_numpy()
        ket_np = ket.to_numpy()

        H.evolve(ket, t=t, result=bra)
        self.assertLess(np.abs(1 - bra.norm()), 1E-9)
        bra_check = bra.to_numpy()

        if ket_np is not None:
            bra_np = linalg.expm_multiply(-1j*t*H_np, ket_np)
            self.assertLess(np.abs(1 - bra_check.dot(bra_np.conj())), 1E-9)

@ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
class Hamiltonians(EvolveChecker):

    def evolve_all(self, t, skip=None):
        if skip is None:
            skip = set()

        for H_name in hamiltonians.get_names():
            if H_name in skip:
                continue

            with self.subTest(H = H_name):
                H = getattr(hamiltonians, H_name)()
                self.evolve_check(H, t)

    def test_zero(self):
        self.evolve_all(0.0)

    def test_short(self):
        self.evolve_all(0.1)

    def test_long(self):
        # otherwise this takes forever
        old_L = config.L
        config.L = max(4, config.L - 4)
        self.evolve_all(50.0, skip={'syk'})
        config.L = old_L

@ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
class ParityTests(EvolveChecker):

    def test_exceptions(self):
        H = identity()
        full_state = State(state=0)
        sub_state = State(state=0, subspace=Parity('even'))

        H.evolve(full_state, t=1.0)
        with self.assertRaises(ValueError):
            H.evolve(sub_state, t=1.0)

        H.add_subspace(Parity('even'))
        H.evolve(sub_state, t=1.0)
        H.evolve(full_state, t=1.0)

    # TODO: actually check output


class Arguments(dtr.DynamiteTestCase):
    def test_unknown_arg(self):
        H = identity()
        state = State(state=0)
        with self.assertRaises(TypeError):
            H.evolve(state, t=1.0, not_valid_arg=True)


@ut.skipIf(complex_enabled(), 'complex numbers enabled')
class RealBuildFail(dtr.DynamiteTestCase):
    def test_fail(self):
        H = identity()
        full_state = State(state=0)
        with self.assertRaises(ValueError):
            H.evolve(full_state, t=1.0)

# TODO: check imaginary time evolution

if __name__ == '__main__':
    dtr.main()
