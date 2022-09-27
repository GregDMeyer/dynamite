
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
from dynamite.computations import MaxIterationsError


class Checker(dtr.DynamiteTestCase):

    def is_close(self, x, y, rtol = 1E-10, atol = 1E-10):
        self.assertTrue(np.isclose(x, y, rtol = rtol, atol = atol),
                        msg = '\n%s\n%s' % (str(x), str(y)))

    def check_all(self, H, evals, evecs, tol=1E-15, evec_tol=None):
        for val, vec in zip(evals, evecs):
            self.check_is_evec(H, vec, val, tol, evec_tol)

        # check that eigenvectors are orthogonal
        for ev1, ev2 in itertools.combinations(evecs, 2):
            self.assertLess(np.abs(ev1.dot(ev2)), tol)

    def check_is_evec(self, H, vec, val, tol=1E-10, evec_tol=None):
        '''
        Check if evec is an eigenvector of H with eigenvalue eval,
        by computing two quantities:

         - evec*H*evec - eval < tol
         - || (1/eval)*H*evec - evec || < tol

        Parameters
        ----------
        H : dynamite.operators.Operator
            The operator

        vec : dynamite.states.State
            The eigenstate

        val : float
            The eigenvalue

        tol : float, optional
            The tolerance for the above comparison

        evec_tol : float, optional
            The tolerance for eigenvectors. Defaults to tol if not set
        '''
        self.assertTrue(isinstance(vec, State))

        if evec_tol is None:
            evec_tol = tol

        prod = H*vec
        self.assertLess(
            abs(val - vec.dot(prod)),
            max(tol, abs(val)*tol)  # max of relative and absolute error
        )

        # apparently binary operators don't work very well in petsc4py
        # so I have to do this manually
        # self.assertLess(((H*vec).vec - val*vec.vec).norm(), tol)
        from dynamite import config
        config._initialize()
        from petsc4py import PETSc

        prod.vec.scale(1/val)

        istart, iend = vec.vec.getOwnershipRange()
        local_prod = np.array(prod.vec[istart:iend])
        local_evec = np.array(vec.vec[istart:iend])

        local_norm = np.linalg.norm(local_prod-local_evec)

        if PETSc.COMM_WORLD.size == 1:
            norm = local_norm
        else:
            CW = PETSc.COMM_WORLD.tompi4py()
            norm = CW.allreduce(local_norm)

        self.assertLess(norm, evec_tol)

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
        for H_name in hamiltonians.get_names(complex_enabled()):
            with self.subTest(H=H_name):
                H = getattr(hamiltonians, H_name)()

                with self.subTest(which='smallest'):
                    evals, evecs = H.eigsolve(nev=5, getvecs=True, tol=1E-12)
                    self.check_all(H, evals, evecs, tol=1E-12, evec_tol=1E-11)

    def test_all_target(self):
        if config.shell:
            self.skipTest("solving for target not supported with shell matrices")

        for H_name in hamiltonians.get_names(complex_enabled()):
            with self.subTest(H=H_name):
                H = getattr(hamiltonians, H_name)()

                lowest_eval = H.eigsolve(which='smallest')[0]
                highest_eval = H.eigsolve(which='largest')[0]

                self.assertLess(lowest_eval, highest_eval)

                for rel_target in [0.25, 0.75]:
                    target = lowest_eval + rel_target*(highest_eval-lowest_eval)
                    with self.subTest(target=target):
                        evals, evecs = H.eigsolve(nev=5, getvecs=True, tol=1E-12, target=target)
                        self.check_all(H, evals, evecs, tol=1E-11, evec_tol=1E-10)

class ZeroDiagonal(Checker):

    def test_smallest(self):
        H = op_sum(0.1*i*sigmax(i) for i in range(config.L))
        evals, evecs = H.eigsolve(nev=5, getvecs=True, tol=1E-12)
        self.check_all(H, evals, evecs, tol=1E-11, evec_tol=1E-9)

    def test_target(self):
        if config.shell:
            self.skipTest("solving for target not supported with shell matrices")

        # coefficients that aren't commensurate but also not random for
        # repeatability
        H = op_sum(np.sin(i)*sigmax(i) for i in range(config.L))
        for target in [0.011, 0.999]:
            with self.subTest(target=target):
                evals, evecs = H.eigsolve(nev=5, getvecs=True, tol=1E-12, target=target)
                self.check_all(H, evals, evecs, tol=1E-11, evec_tol=1E-9)

class ParityTests(Checker):

    def test_exceptions(self):
        H = identity()
        s = Parity('even')

        H.eigsolve()
        with self.assertRaises(ValueError):
            H.eigsolve(subspace=s)

        H.add_subspace(s)

        H.eigsolve()
        H.eigsolve(subspace=s)


class ConvergenceFail(dtr.DynamiteTestCase):

    def test_iterations(self):
        H = hamiltonians.localized()

        eigs = H.eigsolve(nev=1)
        self.assertGreaterEqual(len(eigs), 1)

        # make sure that setting iterations to 2 causes failure
        with self.assertRaises(MaxIterationsError):
            H.eigsolve(nev=1, max_its=2)


if __name__ == '__main__':
    dtr.main()
