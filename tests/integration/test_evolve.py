
import numpy as np
from scipy.sparse import linalg
import hamiltonians

import dynamite_test_runner as dtr
import unittest as ut

from dynamite import config
from dynamite.operators import sigmax, index_product, identity
from dynamite.states import State
from dynamite.subspaces import Parity, SpinConserve, Auto, XParity
from dynamite.tools import complex_enabled
from dynamite.computations import MaxIterationsError


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

class EvolveChecker(dtr.DynamiteTestCase):
    def evolve_check(self, H, t, **kwargs):
        bra, ket = H.create_states()
        ket.set_random(seed = 0)

        H_np = H.to_numpy()
        ket_np = ket.to_numpy()

        H.evolve(ket, t=t, result=bra, **kwargs)

        if t.imag == 0:
            self.assertLess(np.abs(1 - bra.norm()), 1E-9)

        bra_check = bra.to_numpy()

        if ket_np is not None:
            bra_np = linalg.expm_multiply(-1j*t*H_np, ket_np)
            inner_prod = bra_check.dot(bra_np.conj())
            norm = bra.norm()
            self.assertLess(np.abs(1 - (inner_prod/(norm**2))), 1E-9,
                            msg=f'inner prod:{inner_prod}; norm^2:{norm**2}')


class Hamiltonians(EvolveChecker):

    def evolve_all(self, t, skip=None, **kwargs):
        if skip is None:
            skip = set()

        for H_name in hamiltonians.get_names(include_complex=complex_enabled()):
            if H_name in skip:
                continue
            if H_name == 'syk' and self.skip_flags['small_only']:
                continue

            evolve_types = []
            if complex_enabled():
                # since there is already a factor of i in
                # the time evolution exponent, counterintuitively
                # imaginary time evolution is the only one you can
                # do without complex numbers
                evolve_types += ['real']

            if t < 20:  # imaginary doesn't converge for huge t
                evolve_types += ['imaginary']

            for evolve_type in evolve_types:
                with self.subTest(H=H_name, evolve_type=evolve_type):
                    t_factor = {
                        'real': 1,
                        'imaginary': -1j,
                    }[evolve_type]
                    H = getattr(hamiltonians, H_name)()
                    self.evolve_check(H, t*t_factor, **kwargs)

    def test_zero(self):
        if self.skip_flags['medium_only']:
            skip = {'long_range', 'localized', 'syk'}
        else:
            skip = {}

        self.evolve_all(0.0, skip=skip)

    def test_short(self):
        if self.skip_flags['medium_only']:
            skip = {'long_range', 'localized', 'syk'}
        else:
            skip = {}

        self.evolve_all(0.1, skip=skip)

    def test_long(self):
        # skip all hamiltonians for this test on medium-only
        self.skip_on_flag('medium_only')

        # on small_only, only skip syk
        if self.skip_flags['small_only']:
            skip = {'syk'}
        else:
            skip = {}

        # otherwise just skip syk
        self.evolve_all(50.0, skip=skip, max_its=750)

@ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
class Subspaces(EvolveChecker):

    def test_all_subspaces(self):
        skip = set()
        if self.skip_flags['small_only']:
            skip.add('syk')
        if self.skip_flags['medium_only']:
            skip.add('long_range')
            skip.add('localized')

        for H_name in hamiltonians.get_names(include_complex=complex_enabled()):
            if H_name in skip:
                continue

            H = getattr(hamiltonians, H_name)()

            subspaces = [
                Parity('even'),
                Parity('odd'),
                SpinConserve(config.L, config.L//2),
            ]
            all_subspaces = subspaces.copy()

            if config.L % 2 == 0:
                for subspace in subspaces:
                    all_subspaces.append(XParity(subspace, '+'))
                    all_subspaces.append(XParity(subspace, '-'))

            all_subspaces += [
                XParity(sector='+'),
                XParity(sector='-'),
                Auto(H, (1 << (H.L//2)), sort=True),
                Auto(H, (1 << (H.L//2)), sort=False),
            ]

            for subspace in subspaces:
                with self.subTest(subspace=subspace):
                    H.add_subspace(subspace)
                    H.allow_projection = True
                    self.evolve_check(H, 0.1)

    def test_parity_exceptions(self):
        H = identity()
        full_state = State(state=0)
        sub_state = State(state=0, subspace=Parity('even'))

        H.evolve(full_state, t=1.0)
        with self.assertRaises(ValueError):
            H.evolve(sub_state, t=1.0)

        H.add_subspace(Parity('even'))
        H.evolve(sub_state, t=1.0)
        H.evolve(full_state, t=1.0)


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


@ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
class EvolveFail(dtr.DynamiteTestCase):
    def test_evolve_fail(self):
        H = hamiltonians.localized()
        bra, ket = H.create_states()
        ket.set_random(seed=0)

        with self.assertRaises(MaxIterationsError):
            H.evolve(ket, t=10, result=bra, max_its=2)


if __name__ == '__main__':
    dtr.main()
