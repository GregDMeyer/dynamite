
import argparse
import numpy as np

import mpi_test_runner as mtr

class DynamiteTestCase(mtr.MPITestCase):

    def check_vec_equal(self, a, b, eps=None):
        '''
        Compare two PETSc vectors, checking that they are equal.
        '''
        # compare via dot product
        nrm = (a.vec-b.vec).norm()

        # compare the local portions of the vectors
        istart, iend = a.vec.getOwnershipRange()

        if istart == iend:
            return

        a = a.vec[istart:iend]
        b = b.vec[istart:iend]

        # this is the amount of machine rounding error we can accumulate
        if eps is None:
            eps = np.finfo(a.dtype).eps

        diff = np.abs(a-b)
        max_idx = np.argmax(diff)
        far_idxs = np.nonzero(~np.isclose(a, b, rtol=0, atol=eps))[0]
        self.assertTrue(far_idxs.size == 0,
                        msg = '''
{nfar} values do not match.
indices: {idxs}
diff norm: {nrm}
largest diff:
a: {a_real}+i{a_imag}
b: {b_real}+i{b_imag}
diff: {diff}
at {max_idx}'''.format(
    nrm = nrm,
    idxs = far_idxs,
    nfar = far_idxs.size,
    a_real = a[max_idx].real,
    a_imag = a[max_idx].imag,
    b_real = b[max_idx].real,
    b_imag = b[max_idx].imag,
    diff = np.abs(a[max_idx]-b[max_idx]),
    max_idx = max_idx,
))

# add these attributes to the test case
# checks = [
#     ('Equal', operator.eq),
#     ('Less', operator.lt),
#     ('ArrayEqual', np.array_equal),
#     ('True', bool)
# ]
#
# for name, fn in checks:
#     def tmp(*args, msg):
#         result =
#     setattr(MPITestCase, 'mpiAssert'+name, tmp)


def parse_command_line(cmd_argv=None):

    parser = argparse.ArgumentParser(description='Run dynamite integration tests.')

    parser.add_argument('name', nargs='?', default=None,
                        help='Glob expression to specify specific test cases')

    parser.add_argument('-f', '--failfast', action='store_true',
                        help='Stop the tests on first failure')

    parser.add_argument('-v', '--verbose', choices=[0, 1, 2], default=1, type=int,
                        help='Level of detail to show')

    parser.add_argument('-L', type=int, default=10,
                        help='Spin chain length at which to run tests')

    parser.add_argument('--gpu', action='store_true',
                        help='Run the tests on a GPU')

    parser.add_argument('--shell', action='store_true',
                        help='Run the tests using shell matrices')

    parser.add_argument('--slepc-args', type=lambda s: s.strip().split(' '),
                        help='Arguments to pass to SLEPc initialization')

    parser.add_argument('--skip-small', action='store_true',
                        help='Skip tests that are marked as being only for '
                             'small L.')

    parser.add_argument('--skip-medium', action='store_true',
                        help='Skip tests that are marked as being only for '
                             'small or moderate L.')

    return parser.parse_args(cmd_argv)


def main(slepc_args=None):
    from dynamite import config
    args = parse_command_line()

    config.L = args.L
    config.shell = args.shell

    if slepc_args is None:
        slepc_args = []

    if args.slepc_args is not None:
        slepc_args += args.slepc_args

    config.initialize(slepc_args, gpu=args.gpu)

    skip_flags = {
        'small_only': False,
        'medium_only': False,
    }

    if args.skip_small or args.skip_medium:
        skip_flags['small_only'] = True,

    if args.skip_medium:
        skip_flags['medium_only'] = True,

    mtr.main(name=args.name, failfast=args.failfast, verbose=args.verbose, skip_flags=skip_flags)
