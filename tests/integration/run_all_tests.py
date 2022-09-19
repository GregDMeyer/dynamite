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
                        help='Command to launch an MPI job. Set to empty'
                        'string to not use MPI.')

    parser.add_argument('--nprocs', default=None,
                        type=lambda l: [int(x) for x in l.split(',')],
                        help='Comma separated list of number of processors to test with')

    parser.add_argument('-t', '--timeout', default=None, type=float,
                        help='Number of seconds after which tests should time out')

    parser.add_argument('--gpu', action='store_true',
                        help='Run tests using a GPU')

    parser.add_argument('-L', type=int, default=10,
                        help='Set spin chain length for tests of variable size')

    parser.add_argument('-v', type=int, default=0,
                        help='Verbosity of output from each test run')

    parser.add_argument('--skip-small', action='store_true',
                        help='Skip tests that are marked as being only for '
                             'small L.')

    parser.add_argument('--skip-medium', action='store_true',
                        help='Skip tests that are marked as being only for '
                             'small or moderate L.')

    # for compatibility with Python < 3.9
    if hasattr(argparse, 'BooleanOptionalAction'):
        parser.add_argument('--shell', action=argparse.BooleanOptionalAction,
                            help='Whether to run the tests using shell matrices. '
                            'If omitted, tests are repeated with shell on and off')

    args = parser.parse_args(cmd_argv)

    return args

def run_test(mpiexec, nproc, fname, options, timeout=None):
    if mpiexec:
        cmd = mpiexec.split(' ')
        cmd += ['-n', str(nproc)]
    else:
        cmd = []
        if nproc > 1:
            raise ValueError('Cannot run with nproc > 1 without MPI')

    cmd += ['python3', fname] + options
    print(' '.join(cmd))

    try:
        result = run(cmd, stderr=PIPE, timeout=timeout)
        print(result.stderr.decode('UTF-8'))
    except TimeoutExpired as e:
        print('Tests timed out. stderr so far:')
        print(e.stderr.decode('UTF-8'))

    print()

def main():
    fnames = sorted(glob('test_*.py'))

    params = parse_command_line()

    if params.nprocs is None:
        if params.gpu or not params.mpiexec:
            params.nprocs = [1]
        else:
            params.nprocs = [1,3,4]

    const_options = ['-v', str(params.v), '-L', str(params.L)]

    if not hasattr(params, 'shell') or params.shell is None:
        run_options = [[], ['--shell']]
    elif params.shell:
        run_options = [['--shell']]
    else:
        run_options = [[]]

    if params.gpu:
        for opts in run_options:
            opts.append('--gpu')

    if params.skip_small:
        for opts in run_options:
            opts.append('--skip-small')

    if params.skip_medium:
        for opts in run_options:
            opts.append('--skip-medium')

    for fname in fnames:
        for options in run_options:
            for nproc in params.nprocs:
                run_test(params.mpiexec, nproc, fname, options+const_options,
                         timeout=params.timeout)

if __name__ == '__main__':
    main()
