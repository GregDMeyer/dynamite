
import unittest as ut
import sys
from time import sleep
from timeit import default_timer
import operator

class MPITestCase(ut.TestCase):
    '''
    Test class specifically for MPI tests. Currently just makes the output
    prettier to be more compatible with the test runners below.
    '''

    def __str__(self):
        class_name = type(self).__name__
        method_name = self._testMethodName
        return class_name + '.' + method_name

class MPITestRunner:
    '''
    A test runner designed to run well under several MPI processes.
    '''

    def __init__(self, name=None, failfast=False, verbose=1, module=None):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD

        if module is None:
            module = sys.modules['__main__']

        self.failfast = failfast
        self.verbose = verbose

        if name is not None:
            self.suite = ut.defaultTestLoader.loadTestsFromName(name, module)
        else:
            self.suite = ut.defaultTestLoader.loadTestsFromModule(module)

    def run(self):
        result = MPITestResult(comm=self.comm, failfast=self.failfast, verbose=self.verbose)
        result.startTestRun()
        self.suite.run(result)
        result.stopTestRun()

class MPITestResult(ut.TestResult):
    '''
    Keep track of test results from several MPI processes, and print them in a nice
    human-readable format.

    Parameters
    ----------
    comm : mpi4py communicator, optional
        Defaults to COMM_WORLD

    failfast : bool, optional
        Whether to stop on the first failed test

    verbose : int, optional
        How verbose the output should be. Currently valid choices are 0, 1, 2.
    '''

    def __init__(self, comm=None, failfast=False, verbose=1):
        ut.TestResult.__init__(self)

        verbose_options = [0,1,2]
        if verbose not in verbose_options:
            raise ValueError('Valid options for verbose flag are '
                             ','.join(str(x) for x in verbose_options))

        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm

        name_length = 40
        self.headers = ["Test name".ljust(name_length-4) + "Rank:"]
        self.headers += [str(x) for x in range(self.comm.size)]

        self.cur_failures = []
        self.all_problems = []

        self.verbose = verbose
        self.failfast = failfast
        self.start_time = None

    def _print(self, *args, rank=None, verbose=None, **kwargs):

        if rank is not None and not isinstance(rank, list):
            rank = [rank]
        if rank is not None and self.comm.rank not in rank:
            return

        if verbose is not None and not isinstance(verbose, list):
            verbose = [verbose]
        if verbose is not None and self.verbose not in verbose:
            return

        print(*args, file=sys.stderr, **kwargs)
        sys.stderr.flush()
        sleep(0.001)

    def startTestRun(self):
        ut.TestResult.startTestRun(self)
        self._print(*self.headers, rank=0, verbose=[1,2])
        self._print(               rank=0, verbose=[1,2])

        self.start_time = default_timer()

    def stopTestRun(self):
        ut.TestResult.stopTestRun(self)

        self._print(rank=0, verbose=2)
        for test, msg in self.all_problems:
            self._print('----------', rank=0, verbose=2)
            self._print(test, '\n',   rank=0, verbose=2)
            self._print(msg,          rank=0, verbose=2)

        duration =  default_timer() - self.start_time

        self._print(rank=0)
        self._print("{} test{} run in {:.3f} s".format(
            self.testsRun,
            's' if self.testsRun != 1 else '',
            duration), rank=0)

    def startTest(self, test):
        ut.TestResult.startTest(self, test)
        self.cur_failures = []
        self._print(str(test).ljust(len(self.headers[0])), end=' ',
                    rank=0, verbose=[1,2])
        self.comm.barrier()

    def _process_per_rank_errors(self):
        for i in range(self.comm.size):
            self.comm.barrier()
            if self.comm.rank == i:
                msg = ''
                if not self.cur_failures:
                    self._print('.', end=' ', verbose=[1,2])
                else:
                    self._print(self.cur_failures[0][0][0], end=' ', verbose=[1,2])
                    for problem, fail_test, fail_msg in self.cur_failures:
                        msg += "Rank {} {}:".format(self.comm.rank, problem)

                        if hasattr(fail_test, '_subDescription'):
                            msg += " {}".format(fail_test._subDescription())

                        msg += "\n"
                        msg += fail_msg
        return msg

    def _collect_and_print_errors(self, test, msg):
        success = all(f[0] == 'XPECTEDFAIL' for f in self.cur_failures)
        all_success = self.comm.allreduce(success, operator.and_)
        if not all_success:
            # send all error messages to process 0
            if not self.cur_failures:
                problem = None
            else:
                problem = self.cur_failures[0][0]

            problems = self.comm.gather(problem, root=0)
            msgs = self.comm.gather(msg, root=0)
            if self.comm.rank == 0:
                real_problems = [p for p in problems if p is not None]
                self.all_problems.append((test, '\n'.join(m for m in msgs if m)))
            else:
                real_problems = ['N'] # placeholder on other procs

            if self.failfast:
                self.stop()

            self._print(real_problems[0][0], end='', rank=0, verbose=0)
        else:
            self._print('.', end='', rank=0, verbose=0)

    def stopTest(self, test):
        ut.TestResult.stopTest(self, test)

        msg = self._process_per_rank_errors()

        self.comm.barrier()
        self._print(rank=0, verbose=[1,2])

        self._collect_and_print_errors(test, msg)

        self.comm.barrier()

    def addError(self, test, err):
        self.cur_failures.append(('ERROR', test, self._exc_info_to_string(err, test)))

    def addFailure(self, test, err):
        self.cur_failures.append(('FAILED', test, self._exc_info_to_string(err, test)))

    def addSubTest(self, test, subtest, err):
        if err is not None:
            if self.failfast:
                self.stop()
            if issubclass(err[0], test.failureException):
                err_str = 'FAILED'
            else:
                err_str = 'ERROR'
            self.cur_failures.append((err_str, subtest, self._exc_info_to_string(err, subtest)))

    def addSkip(self, test, reason):
        self.cur_failures.append(('SKIPPED', test, reason))

    def addExpectedFailure(self, test, err):
        self.cur_failures.append(('XPECTEDFAIL', test, self._exc_info_to_string(err, test)))

    def addUnexpectedSuccess(self, test):
        self.cur_failures.append(('UNEXPECTSUCCESS', test, ''))

def main(name=None, failfast=False, verbose=1, module=None):
    '''
    Run tests.

    Parameters
    ----------
    name : str or None, optional
        A string specifying a subset of tests to run, or None to run all tests

    failfast : bool, optional
        Whether to stop on the first failed test

    verbose : int, optional
        How verbose to make the output

    module : module or None
        Module in which to find the tests. Defaults to __main__ if None.
    '''
    runner = MPITestRunner(name=name, failfast=failfast, verbose=verbose, module=module)
    runner.run()
