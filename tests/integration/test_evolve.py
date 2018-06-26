
import unittest as ut
import numpy as np
from scipy import linalg
import hamiltonians

from dynamite import config
from dynamite.operators import sigmax, index_product

config.L = 8
config.initialize()

class Analytic(ut.TestCase):
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

class Hamiltonians(ut.TestCase):

    def evolve_all(self, t):
        for H_name in hamiltonians.__all__:
            with self.subTest(H = H_name):
                H = getattr(hamiltonians, H_name)()
                bra, ket = H.create_states()
                ket.set_random(seed = 0)

                H_np = H.to_numpy()
                ket_np = ket.to_numpy()

                H.evolve(ket, t = t, result = bra)
                self.assertLess(np.abs(1 - bra.norm()), 1E-9)
                bra_check = bra.to_numpy()

                if ket_np is not None:
                    bra_np = linalg.expm(-1j*t*H_np).dot(ket_np)
                    self.assertLess(np.abs(1 - bra_check.dot(bra_np.conj())), 1E-9)

    def test_zero(self):
        self.evolve_all(0.0)

    def test_short(self):
        self.evolve_all(0.1)

    def test_long(self):
        self.evolve_all(10.0)

if __name__ == '__main__':
    ut.main()
