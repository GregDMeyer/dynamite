
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
from dynamite.computations import MaxIterationsError


@ut.skipIf(not complex_enabled(), 'complex numbers not enabled')
class EvolveFail(dtr.DynamiteTestCase):
    def test_evolve_fail(self):
        H = hamiltonians.localized()
        bra, ket = H.create_states()
        ket.set_random(seed=0)

        with self.assertRaises(MaxIterationsError):
            H.evolve(ket, t=10, result=bra)


if __name__ == '__main__':
    # set the max iterations to be small
    dtr.main(['-mfn_max_it', '1'])
