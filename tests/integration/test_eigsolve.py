
import unittest as ut
import itertools
import numpy as np
import hamiltonians

from dynamite.operators import index_sum, sigmaz, sigmax
from dynamite.states import State

class Checker(ut.TestCase):

    def is_close(self, x, y, rtol = 1E-10, atol = 1E-10):
        self.assertTrue(np.isclose(x, y, rtol = rtol, atol = atol),
                        msg = '\n%s\n%s' % (str(x), str(y)))

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
            evals, evecs = H.eigsolve(nev = 1 + H.get_length(),
                                      getvecs = True,
                                      which = 'smallest')
            self.is_close(evals[0], -H.get_length())
            for i in range(1, H.get_length()+1):
                with self.subTest(eval_idx = i):
                    self.is_close(evals[i], -H.get_length()+2)

        with self.subTest(which = 'largest'):
            evals, evecs = H.eigsolve(nev = 1 + H.get_length(),
                                      getvecs = True,
                                      which = 'largest')
            self.is_close(evals[0], H.get_length())
            for i in range(1, H.get_length()+1):
                with self.subTest(eval_idx = i):
                    self.is_close(evals[i], H.get_length()-2)

class Hamiltonians(Checker):

    def check_all(self, H, evals, evecs, tol = 1E-15):
        for val, vec in zip(evals, evecs):
            self.check_is_evec(H, vec, val, tol)

        # check that eigenvectors are orthogonal
        for ev1, ev2 in itertools.combinations(evecs, 2):
            self.assertLess(np.abs(ev1.dot(ev2)), tol)

    def test_all(self):
        for H_name in hamiltonians.__all__:
            with self.subTest(H = H_name):
                H = getattr(hamiltonians, H_name)()

                with self.subTest(which = 'smallest'):
                    evals, evecs = H.eigsolve(nev = 5, getvecs = True, tol = 1E-12)
                    self.check_all(H, evals, evecs, tol = 1E-11)

if __name__ == '__main__':
    from dynamite import config
    config.L = 14
    ut.main()
