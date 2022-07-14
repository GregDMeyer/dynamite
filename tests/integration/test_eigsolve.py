
import unittest as ut
import itertools
import numpy as np
import hamiltonians

import dynamite_test_runner as dtr

from dynamite import config
from dynamite.operators import index_sum, sigmax, identity, op_sum
from dynamite.states import State
from dynamite.subspaces import Parity
from dynamite.tools import complex_enabled

class Checker(dtr.DynamiteTestCase):

    def is_close(self, x, y, rtol = 1E-10, atol = 1E-10):
        self.assertTrue(np.isclose(x, y, rtol = rtol, atol = atol),
                        msg = '\n%s\n%s' % (str(x), str(y)))

    def check_all(self, H, evals, evecs, tol = 1E-15):
        for val, vec in zip(evals, evecs):
            self.check_is_evec(H, vec, val, tol)

        # check that eigenvectors are orthogonal
        for ev1, ev2 in itertools.combinations(evecs, 2):
            self.assertLess(np.abs(ev1.dot(ev2)), tol)

    def check_is_evec(self, H, vec, val, tol = 1E-10):
        '''
        Check if evec is an eigenvector of H with eigenvalue eval,
        by computing for normalized eigenvector:
            norm(H*evec - eval*evec) < tol

        Parameters
        ----------
        H : dynamite.operators.Operator
            The operator

        vec : dynamite.states.State
            The eigenstate

        val : float
            The eigenvalue

        tol : float
            The tolerance for the above comparison
        '''
        self.assertTrue(isinstance(vec, State))

        # apparently binary operators don't work very well in petsc4py
        # so I have to do this manually
        # self.assertLess(((H*vec).vec - val*vec.vec).norm(), tol)

        # the norm doesn't depend on other processes values, so we can just
        # assert independently
        istart, iend = vec.vec.getOwnershipRange()
        local_prod = np.array((H*vec).vec[istart:iend])
        local_evec = val*np.array(vec.vec[istart:iend])
        self.assertLess(np.linalg.norm(local_prod-local_evec), tol)

class Analytic(Checker):
    '''
    Some cases in which we can easily write down the answer.
    '''

    def test_evals_only(self):
        H = index_sum(sigmax())
        evals = H.eigsolve()
        self.is_close(evals[0], -H.get_length())

    def test_uniform_field(self):
        H = index_sum(sigmax())

        # TODO: check eigenvectors here

        with self.subTest(which = 'smallest'):
            evals, evecs = H.eigsolve(nev = 2,
                                      getvecs = True,
                                      which = 'smallest')
            self.is_close(evals[0], -H.get_length())
            self.is_close(evals[1], -H.get_length() + 2)

        with self.subTest(which = 'largest'):
            evals, evecs = H.eigsolve(nev = 2,
                                      getvecs = True,
                                      which = 'largest')
            self.is_close(evals[0], H.get_length())
            self.is_close(evals[1], H.get_length() - 2)

class Hamiltonians(Checker):

    def test_all_smallest(self):
        for H_name, real in hamiltonians.names:
            if not complex_enabled() and not real:
                continue
            with self.subTest(H=H_name):
                H = getattr(hamiltonians, H_name)()

                with self.subTest(which = 'smallest'):
                    evals, evecs = H.eigsolve(nev = 5, getvecs = True, tol = 1E-12)
                    self.check_all(H, evals, evecs, tol = 1E-10)

    def test_all_target(self):
        if config.shell:
            self.skipTest("solving for target not supported with shell matrices")

        for H_name, real in hamiltonians.names:
            if not complex_enabled() and not real:
                continue
            with self.subTest(H=H_name):
                H = getattr(hamiltonians, H_name)()

                for target in [0.01, 0.9]:
                    with self.subTest(target=target):
                        evals, evecs = H.eigsolve(nev=5, getvecs=True, tol=1E-12, target=target)
                        self.check_all(H, evals, evecs, tol=1E-9)

class ZeroDiagonal(Checker):

    def test_smallest(self):
        H = op_sum(0.1*i*sigmax(i) for i in range(config.L))
        evals, evecs = H.eigsolve(nev=5, getvecs=True, tol=1E-12)
        self.check_all(H, evals, evecs, tol=1E-10)

    def test_target(self):
        if config.shell:
            self.skipTest("solving for target not supported with shell matrices")

        H = op_sum(0.1*i*sigmax(i) for i in range(config.L))
        for target in [0.011, 0.999]:
            with self.subTest(target=target):
                evals, evecs = H.eigsolve(nev=5, getvecs=True, tol=1E-12, target=target)
                self.check_all(H, evals, evecs, tol=1E-9)

class ParityTests(Checker):

    def test_exceptions(self):
        H = identity()
        s = Parity('even')
        s.L = H.L
        # TODO: automatically set L somehow?

        H.eigsolve()
        with self.assertRaises(ValueError):
            H.eigsolve(subspace=s)

        H.add_subspace(s)

        H.eigsolve()
        H.eigsolve(subspace=s)

    # TODO: actually check results

if __name__ == '__main__':
    dtr.main()
