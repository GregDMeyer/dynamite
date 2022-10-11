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
from dynamite.operators import index_product, sigma_plus, sigma_minus


def collision_operator():
    '''
    An extremely contrived operator where terms collide for spinflip.
    '''
    half = 2*(config.L//4)
    left_half = 0.75*index_product(sigma_plus(), size=half//2)
    left_half *= 0.75*index_product(sigma_minus(half//2), size=half)

    left_half_2 = 0.75*index_product(sigma_minus(), size=half//2)
    left_half_2 *= 0.75*index_product(sigma_plus(half//2), size=half)

    right_half = -0.25*index_product(sigma_plus(half), size=3*half//2)
    right_half *= -0.25*index_product(sigma_minus(3*half//2))

    right_half_2 = -0.25*index_product(sigma_minus(half), size=3*half//2)
    right_half_2 *= -0.25*index_product(sigma_plus(3*half//2))

    return left_half+left_half_2+right_half+right_half_2


class Hamiltonians(dtr.DynamiteTestCase):

    def test_norm(self):
        all_H = hamiltonians.get_names(complex_enabled()) + ['collision']
        for H_name in all_H:
            if H_name == 'syk' and self.skip_flags['small_only']:
                continue
            test_subspaces = ('full', 'spinconserve', 'mix', 'spinflip+', 'spinflip-')
            for subspace_name in test_subspaces:
                with self.subTest(H=H_name, subspace=subspace_name):
                    if H_name == 'collision':
                        H = collision_operator()
                    else:
                        H = getattr(hamiltonians, H_name)()

                    if subspace_name == 'spinconserve':
                        H.subspace = SpinConserve(config.L, config.L//3)
                        H.allow_projection = True
                    elif subspace_name.startswith('spinflip'):
                        if config.L % 2 == 1:
                            continue
                        H.subspace = SpinConserve(
                            config.L, config.L//2, spinflip=subspace_name[-1]
                        )
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
