'''
Test correctness of matrix norm for shell matrices.
'''

import unittest as ut
import numpy as np
import hamiltonians

import dynamite_test_runner as dtr

from dynamite import config
from dynamite._backend.bbuild import have_gpu_shell

class Hamiltonians(dtr.DynamiteTestCase):

    def do_all_shell(self, shelltype):
        config.initialize()
        from petsc4py import PETSc

        for H_name in hamiltonians.__all__:
            with self.subTest(H = H_name):
                H = getattr(hamiltonians, H_name)()
                H.shell = shelltype
                petsc_norm = H.get_mat().norm(PETSc.NormType.INFINITY)

                H.shell = False
                shell_norm = H.get_mat().norm(PETSc.NormType.INFINITY)

                # the extra factor of 1E2 error is because PETSc ignores rounding
                # errors when they compute their norm!!
                eps = H.nnz * np.finfo(np.complex128).eps * 1E2
                self.assertLess(np.abs(petsc_norm-shell_norm), eps,
                                msg = '\npetsc: %e\nshell: %e' % (petsc_norm, shell_norm))

    def test_all_cpu(self):
        self.do_all_shell('cpu')

    @ut.skipIf(not have_gpu_shell(), reason = 'not built with GPU support')
    def test_all_gpu(self):
        self.do_all_shell('gpu')

# TODO: check correctness in the various subspace combinations

if __name__ == '__main__':
    dtr.main()
