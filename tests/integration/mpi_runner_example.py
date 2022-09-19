
import argparse
import mpi_test_runner as mtr
from mpi4py import MPI


class Example(mtr.MPITestCase):

    def test_pass(self):
        self.assertTrue(True,
                        msg='this test should pass')

    def test_fail(self):
        self.assertTrue(False,
                        msg='this test should fail')

    def test_error(self):
        raise Exception()

    def test_assert_throws_pass(self):
        class CustomError(Exception):
            pass
        with self.assertRaises(CustomError):
            raise CustomError()

    def test_assert_throws_fail(self):
        class CustomError(Exception):
            pass
        with self.assertRaises(CustomError):
            pass

    def test_assert_throws_error(self):
        class CustomError(Exception):
            pass
        with self.assertRaises(CustomError):
            raise Exception()

    def test_skip(self):
        self.skipTest("this test was skipped")

    def test_fail_zero(self):
        self.assertTrue(MPI.COMM_WORLD.rank != 0)

    def test_fail_nonzero(self):
        self.assertTrue(MPI.COMM_WORLD.rank == 0)

    def test_fail_odd(self):
        self.assertTrue(MPI.COMM_WORLD.rank % 2 == 0)

    def test_fail_even(self):
        self.assertTrue(MPI.COMM_WORLD.rank % 2 == 1)

    def test_error_zero(self):
        if MPI.COMM_WORLD.rank == 0:
            raise Exception()

    def test_error_nonzero(self):
        if MPI.COMM_WORLD.rank != 0:
            raise Exception()

    def test_error_odd(self):
        if MPI.COMM_WORLD.rank % 2 != 0:
            raise Exception()

    def test_error_even(self):
        if MPI.COMM_WORLD.rank % 2 == 0:
            raise Exception()

    def test_mix_1(self):
        if MPI.COMM_WORLD.rank % 2 == 0:
            raise Exception()
        else:
            self.assertTrue(False)

    def test_mix_2(self):
        if MPI.COMM_WORLD.rank % 2 != 0:
            raise Exception()
        else:
            self.assertTrue(False)

    def test_skip_flag(self):
        self.skip_on_flag('test_flag')


def parse_command_line():

    parser = argparse.ArgumentParser(description='Test the MPI test runner.')

    parser.add_argument('name', nargs='?', default=None,
                        help='Glob expression to specify specific test cases')

    parser.add_argument('-f', '--failfast', action='store_true',
                        help='Stop the tests on first failure')

    parser.add_argument('-v', '--verbose', choices=[0, 1, 2], default=1,
                        type=int,
                        help='Level of detail to show')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    mtr.main(name=args.name, failfast=args.failfast, verbose=args.verbose, skip_flags={'test_flag': True})
