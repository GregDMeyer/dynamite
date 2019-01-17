
import argparse
import numpy as np

import mpi_test_runner as mtr

class DynamiteTestCase(mtr.MPITestCase):

    def check_vec_equal(self, a, b, eps=None):
        '''
        Compare two PETSc vectors, checking that they are equal.
        '''
        # compare the local portions of the vectors
        istart, iend = a.vec.getOwnershipRange()
        a = a.vec[istart:iend]
        b = b.vec[istart:iend]

        # this is the amount of machine rounding error we can accumulate
        if eps is None:
            eps = np.finfo(a.dtype).eps

        diff = np.abs(a-b)
        max_idx = np.argmax(diff)
        self.assertTrue(np.allclose(a, b, rtol=0, atol=eps),
                        msg = '\na: %e+i%e\nb: %e+i%e\ndiff: %e\nat %d' % \
                            (a[max_idx].real, a[max_idx].imag,
                             b[max_idx].real, b[max_idx].imag,
                             np.abs(a[max_idx]-b[max_idx]), max_idx))

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

    parser.add_argument('--slepc_args', type=lambda s: s.strip().split(' '),
                        help='Arguments to pass to SLEPc initialization')

    return parser.parse_args(cmd_argv)

def main():
    from dynamite import config
    args = parse_command_line()

    config.L = args.L

    if args.shell:
        if args.gpu:
            config.shell = 'gpu'
        else:
            config.shell = 'cpu'

    if args.gpu:
        args.slepc_args += [
            '-vec_type', 'cuda',
            '-mat_type', 'aijcusparse',
        ]

    config.initialize(args.slepc_args)

    mtr.main(name=args.name, failfast=args.failfast, verbose=args.verbose)
