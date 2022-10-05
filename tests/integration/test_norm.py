'''
Test correctness of matrix norm for shell matrices.
'''

import numpy as np
from scipy.sparse.linalg import norm
import hamiltonians

import dynamite_test_runner as dtr

from dynamite import config
from dynamite.tools import complex_enabled
from dynamite.subspaces import SpinConserve, Parity


class Hamiltonians(dtr.DynamiteTestCase):

    def test_norm(self):
        for H_name in hamiltonians.get_names(complex_enabled()):
            if H_name == 'syk' and self.skip_flags['small_only']:
                continue
            for subspace_name in ('full', 'spinconserve', 'mix'):
                with self.subTest(H=H_name, subspace=subspace_name):
                    H = getattr(hamiltonians, H_name)()

                    if subspace_name == 'spinconserve':
                        H.subspace = SpinConserve(config.L, config.L//3)
                        H.allow_projection = True
                    elif subspace_name == 'mix':
                        H.add_subspace(
                            SpinConserve(config.L, config.L//3),
                            Parity('even')
                        )
                        H.allow_projection = True

                    H.shell = True
                    petsc_norm = H.infinity_norm()

                    H.shell = False
                    shell_norm = H.infinity_norm()

                    # the extra factor of 1E2 error is because PETSc ignores
                    # floating point errors when they compute their norm!!
                    eps = H.nnz * np.finfo(np.complex128).eps * 1E2

                    msg = f'\npetsc: {petsc_norm}\nshell: {shell_norm}'
                    self.assertLess(np.abs(petsc_norm-shell_norm), eps,
                                    msg=msg)

                    if not self.skip_flags['medium_only']:
                        numpy_norm = norm(H.to_numpy(), ord=np.inf)
                        msg = f'\nnumpy: {numpy_norm}\nshell: {shell_norm}'
                        self.assertLess(np.abs(numpy_norm-shell_norm), eps,
                                        msg=msg)


if __name__ == '__main__':
    dtr.main()
