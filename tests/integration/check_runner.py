
import mpi_test_runner as mtr
from mpi4py import MPI
import unittest as ut


class Simple(mtr.MPITestCase):

    def test_pass(self):
        pass

    def test_fail(self):
        self.fail()

    def test_fail_0(self):
        if MPI.COMM_WORLD.rank == 0:
            self.fail()

    def test_fail_1(self):
        if MPI.COMM_WORLD.rank == 1:
            self.fail()

    def test_fail_nonzero(self):
        if MPI.COMM_WORLD.rank != 0:
            self.fail()

    def test_error(self):
        raise ValueError

    def test_error_nonzero(self):
        if MPI.COMM_WORLD.rank != 0:
            raise ValueError

    def test_valid_error(self):
        with self.assertRaises(ValueError):
            raise ValueError

    def test_mixed(self):
        if MPI.COMM_WORLD.rank < 2:
            self.fail()
        else:
            raise ValueError

    def test_subtest_fail(self):
        with self.subTest(sub='test'):
            self.fail()

    def test_subtest_double(self):
        with self.subTest(sub='error'):
            raise ValueError
        with self.subTest(sub='fail'):
            self.fail()

    def test_subtest_error(self):
        with self.subTest(sub='test'):
            raise ValueError

    @ut.expectedFailure
    def test_expected_fail(self):
        self.fail()

    @ut.expectedFailure
    def test_unexpected_success(self):
        pass

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Check MPI test runner behavior.')
    parser.add_argument('name', nargs='?', default=None,
                        help='Glob expression to specify specific test cases')
    parser.add_argument('-f', '--failfast', action='store_true',
                        help='Stop the tests on first failure.')
    parser.add_argument('-v', '--verbose', choices=[0, 1, 2], default=1, type=int,
                        help='Level of detail to show')
    args = parser.parse_args()

    mtr.main(args.name, args.failfast, args.verbose)
