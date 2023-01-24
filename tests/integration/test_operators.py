'''
Integration tests for operators.
'''

import numpy as np

import dynamite_test_runner as dtr

from dynamite import config
from dynamite.tools import complex_enabled
from dynamite.msc_tools import msc_dtype
from dynamite.subspaces import Full, Parity, SpinConserve, Auto, XParity
from dynamite.operators import index_sum, sigmax, sigmay, sigmaz
from dynamite.operators import Operator

import hamiltonians


class SubspaceConservation(dtr.DynamiteTestCase):
    """
    Tests ensuring that dynamite correctly checks whether
    a given subspace is conserved or not.
    """

    def test_full(self):
        for H_name in hamiltonians.get_names(complex_enabled()):
            with self.subTest(H=H_name):
                H = getattr(hamiltonians, H_name)()
                self.assertTrue(H.conserves(Full()))

    def test_parity(self):
        answers = {
            'localized': True,
            'syk': True,
            'ising': False,
            'long_range': False
        }

        for parity in ('even', 'odd'):
            for H_name in hamiltonians.get_names(complex_enabled()):
                H = getattr(hamiltonians, H_name)()
                with self.subTest(H=H_name, parity=parity):
                    self.assertEqual(
                        H.conserves(Parity(parity)),
                        answers[H_name]
                    )

    def test_spinconserve(self):
        answers = {
            'localized': True,
            'syk': False,
            'ising': False,
            'long_range': False
        }

        for k in (config.L//2, config.L//4):
            for H_name in hamiltonians.get_names(complex_enabled()):
                H = getattr(hamiltonians, H_name)()
                with self.subTest(H=H_name, L=config.L, k=k):
                    self.assertEqual(
                        H.conserves(SpinConserve(config.L, k)),
                        answers[H_name]
                    )

    def test_spinconserve_xparity_false(self):
        L = config.L

        if L % 2 != 0:
            self.skipTest("only for even spin chain lengths")

        k = config.L//2
        for xparity in ('+', '-'):
            for H_name in hamiltonians.get_names(complex_enabled()):
                H = getattr(hamiltonians, H_name)()
                with self.subTest(H=H_name, xparity=xparity):
                    self.assertFalse(
                        H.conserves(
                            XParity(
                                SpinConserve(config.L, k),
                                sector=xparity
                            )
                        )
                    )

    def test_spinconserve_xparity_heisenberg(self):
        H = index_sum(
            sigmax(0)*sigmax(1) + sigmay(0)*sigmay(1) + sigmaz(0)*sigmaz(1)
        )

        L = config.L
        if L % 2 != 0:
            self.skipTest("only for even spin chain lengths")

        k = config.L//2
        for xparity in ('+', '-'):
            with self.subTest(xparity=xparity):
                self.assertTrue(
                    H.conserves(
                        XParity(
                            SpinConserve(config.L, k),
                            sector=xparity
                        )
                    )
                )

    def test_spinconserve_xparity_error(self):
        op = sigmax()
        with self.assertRaises(ValueError):
            op.conserves(
                XParity(SpinConserve(config.L, config.L//2), sector='+'),
                Full()
            )

    def test_auto(self):
        for k in (config.L//2, config.L//4):
            for H_name in hamiltonians.get_names(complex_enabled()):
                if H_name == 'syk' and self.skip_flags['small_only']:
                    continue
                H = getattr(hamiltonians, H_name)()
                subspace = Auto(H, 'U'*k + 'D'*(config.L-k))
                with self.subTest(H=H_name, L=config.L, k=k):
                    self.assertTrue(
                        H.conserves(subspace)
                    )

    def test_change_parity(self):
        """
        Test going from one parity subspace to the other
        """
        op = sigmax()

        with self.subTest(from_space='odd', to_space='even'):
            self.assertTrue(
                op.conserves(Parity('even'), Parity('odd'))
            )

        with self.subTest(from_space='even', to_space='odd'):
            self.assertTrue(
                op.conserves(Parity('odd'), Parity('even'))
            )

    # Switching the value of xparity is currently not supported, but could be
    # in the future
    # def test_change_xparity(self):
    #     """
    #     Test operators that take us from one xparity value to the other
    #     """
    #     L = config.L
    #     if L % 2 != 0:
    #         self.skipTest("only for even spin chain lengths")
    #     k = config.L//2

    #     op = sigmaz()

    #     self.assertTrue(
    #         op.conserves(
    #             XParity(SpinConserve(config.L, k), sector='+'),
    #             XParity(SpinConserve(config.L, k), sector='-')
    #         )
    #     )

    #     self.assertTrue(
    #         op.conserves(
    #             XParity(SpinConserve(config.L, k), sector='-'),
    #             XParity(SpinConserve(config.L, k), sector='+')
    #         )
    #     )

    def test_full_to_others(self):
        """
        all these subspaces should fail projecting from full space
        """
        subspaces = [
            ('parity_even', Parity('even')),
            ('parity_odd', Parity('odd')),
            ('spinconserve', SpinConserve(config.L, config.L//2))
        ]
        for H_name in hamiltonians.get_names(complex_enabled()):
            with self.subTest(H=H_name):
                H = getattr(hamiltonians, H_name)()
                for subspace_name, subspace in subspaces:
                    with self.subTest(subspace=subspace_name):
                        self.assertFalse(H.conserves(
                            subspace,
                            Full()
                        ))


class SaveLoad(dtr.DynamiteTestCase):

    def test_save_load(self):
        for H_name in hamiltonians.get_names(complex_enabled()):
            if 'slow' in self.skip_flags and H_name == 'syk':
                continue

            with self.subTest(H=H_name):
                H = getattr(hamiltonians, H_name)()

                fname = '/tmp/test_save.dnm'
                H.save(fname)
                H_new = Operator.load(fname)
                self.assertEqual(H, H_new)

    def test_load_int64_fail(self):
        test_string = b'1\n64\n\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        have_int64 = msc_dtype['masks'].itemsize == 8
        if have_int64:
            self.assertEqual(
                Operator.from_bytes(test_string),
                sigmax(33)
            )
        else:
            with self.assertRaises(ValueError):
                Operator.from_bytes(test_string)

    def test_load_int64_success(self):
        test_string = b'1\n64\n\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        self.assertEqual(
            Operator.from_bytes(test_string),
            sigmax(2)
        )


class MSCConsistency(dtr.DynamiteTestCase):

    def setUp(self):
        config._initialize()
        from petsc4py import PETSc
        if PETSc.COMM_WORLD.size == 1:
            self.skipTest(
                "test only applicable for >1 rank"
            )

    def test_index_fail(self):
        config._initialize()
        from petsc4py import PETSc

        # different operator on each rank
        op = sigmax(PETSc.COMM_WORLD.rank % config.L)

        with self.assertRaises(RuntimeError):
            op.build_mat()

    def test_random_coeff_fail(self):
        config._initialize()
        from petsc4py import PETSc

        # different seed on each rank
        rng = np.random.default_rng(
            seed=PETSc.COMM_WORLD.rank
        )

        H = hamiltonians.long_range()
        op = rng.random()*H

        with self.assertRaises(RuntimeError):
            op.build_mat()


class Exceptions(dtr.DynamiteTestCase):

    def test_scale(self):
        op = sigmax()
        with self.assertRaises(TypeError):
            op.scale('hi')


if __name__ == '__main__':
    dtr.main()
