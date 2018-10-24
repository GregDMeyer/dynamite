'''
Run all integration tests for dynamite, on varying numbers of processors and in
various configurations.
'''

from subprocess import run, PIPE, TimeoutExpired
from glob import glob
import argparse

def parse_command_line(cmd_argv=None):
    parser = argparse.ArgumentParser(description='Run all tests for dynamite.')

    parser.add_argument('--mpiexec', default='mpirun',
                        help='Command to launch an MPI job')

    parser.add_argument('--nprocs', default=[1,3,4],
                        type=lambda l: [int(x) for x in l.split(',')],
                        help='Comma separated list of number of processors to test with')

    parser.add_argument('-t', '--timeout', default=None, type=float,
                        help='Number of seconds after which tests should time out')

    parser.add_argument('--gpu', action='store_true',
                        help='Also run tests using a GPU')

    parser.add_argument('-L', type=int, default=10,
                        help='Set spin chain length for tests of variable size')

    args = parser.parse_args(cmd_argv)

    return args

def run_test(mpiexec, nproc, fname, options, timeout=None):
    cmd = [mpiexec, '-n', str(nproc), 'python', fname] + options
    print(' '.join(cmd))

    try:
        result = run(cmd, stderr=PIPE, timeout=timeout)
        print(result.stderr.decode('UTF-8'))
    except TimeoutExpired as e:
        print('Tests timed out. stderr so far:')
        print(e.stderr.decode('UTF-8'))

    print()

def main():
    fnames = sorted(glob('test_*'))

    params = parse_command_line()

    const_options = ['-v', '0', '-L', str(params.L)]
    run_options = [[], ['--shell']]

    if params.gpu:
        run_options += [['--gpu'], ['--shell', '--gpu']]

    for fname in fnames:
        for options in run_options:
            for nproc in params.nprocs:
                run_test(params.mpiexec, nproc, fname, options+const_options,
                         timeout=params.timeout)

if __name__ == '__main__':
    main()
