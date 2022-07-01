"""
This script builds Docker images for various combinations of
parameters. Should be run from inside the git tree.
"""

import sys
from os import path, mkdir, set_blocking, remove
from argparse import ArgumentParser
from subprocess import run, PIPE, Popen
from contextlib import ExitStack
from time import time, sleep
from datetime import timedelta
import shutil
import atexit


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser(argv)

    parser.add_argument("--targets", type=lambda x: x.split(','),
                        default=['jupyter', 'release'],
                        help='Targets to build, delimited by commas.')

    parser.add_argument("--platform", type=lambda x: x.split(','),
                        default=['cpu', 'gpu'],
                        help='Whether to build the CPU and/or GPU versions.')

    parser.add_argument("-v", "--verbose", action='store_true',
                        help='Show build output in terminal. '
                        'Turns off parallel builds.')

    parser.add_argument("--fresh", action='store_true',
                        help='Update cached images.')

    parser.add_argument("--cuda-archs", type=lambda x: x.split(','),
                        default=['70', '80'],
                        help='CUDA compute capabilities.')

    parser.add_argument("--int-sizes", type=lambda x: [int(v) for v in x.split(',')],
                        default=[32, 64],
                        help='Bits in an integer (allowed values: 32 and 64)')

    parser.add_argument("--dry-run", action="store_true",
                        help="Print build commands but do not run them.")

    parser.add_argument("--no-parallel", action="store_true",
                        help="Turn off parallel builds.")

    args = parser.parse_args()

    if args.verbose or args.dry_run:
        args.no_parallel = True

    # I don't know if we actually need a flag for this
    args.log_dir = "/tmp/dnm_build_logs"

    return args


def main():
    args = parse_args()

    build_dir = "/tmp/dnm_docker_build"
    if path.exists(build_dir):
        build_dir += '_'+str(int(time()))

    # remove build files whether we're successful or not
    def remove_build_files(dir_to_remove):
        print("Removing build files...")
        shutil.rmtree(dir_to_remove)
    atexit.register(remove_build_files, build_dir)

    if not path.exists(args.log_dir):
        mkdir(args.log_dir)

    # checkout a clean version to build from
    git_root = run(["git", "rev-parse", "--show-toplevel"],
                   capture_output=True, text=True, check=True).stdout.strip()
    mkdir(build_dir)
    run(["git", "init"], capture_output=True, cwd=build_dir, check=True)
    run(["git", "pull", git_root], capture_output=True,
        cwd=build_dir, check=True)

    # remove git logs and index so they don't invalidate the build cache
    shutil.rmtree(path.join(build_dir, '.git/logs'))
    remove(path.join(build_dir, '.git/index'))

    version = open(path.join(build_dir, 'VERSION')).read().strip()

    start_time = time()

    # run builds
    first_target = True
    for target in args.targets:

        if not first_target:
            # it seems the docker cache needs some time to catch up
            sleep(10)

        builds = []
        for platform in args.platform:
            if platform == 'gpu':
                cuda_archs = args.cuda_archs
            else:
                cuda_archs = [None]

            for cuda_arch in cuda_archs:
                for int_size in args.int_sizes:

                    # this configuration is currently not supported
                    if platform == 'gpu' and int_size == 64:
                        print('Skipping unsupported 64-bit GPU build')
                        continue

                    tags = ["latest", version]

                    if platform == 'gpu':
                        tags = [tag+'-cuda' for tag in tags]
                        no_cc_tags = tags.copy()
                        tags = [tag+'.cc'+cuda_arch for tag in tags]

                        # default is cc 7.0
                        if cuda_arch == '70':
                            tags = no_cc_tags + tags

                    cmd = [
                        "docker", "build",
                        "--build-arg", f"PLATFORM={platform}",
                        "-f", "docker/Dockerfile",
                        "--target", target
                    ]

                    if cuda_arch is not None:
                        cmd += ["--build-arg", f"CUDA_ARCH={cuda_arch}"]

                    if int_size == 64:
                        cmd += ["--build-arg", "PETSC_CONFIG_FLAGS=--with-64-bit-indices"]
                        tags = [tag+'-int64' for tag in tags]
                    elif int_size != 32:
                        raise ValueError(f"Unknown int size '{int_size}'")

                    if args.fresh and first_target:
                        cmd += ["--no-cache", "--pull"]

                    if target == 'jupyter':
                        tags = [tag+'-jupyter' for tag in tags]

                    for tag in tags:
                        cmd += ["-t", f"gdmeyer/dynamite:{tag}"]

                    cmd += ["."]

                    build_dict = {}
                    build_dict['cmd'] = cmd
                    build_dict['tags'] = tags

                    builds.append(build_dict)

        if not args.no_parallel:
            build_parallel(builds, build_dir, args)
        else:
            build_sequential(builds, build_dir, args)

        first_target = False

    elapsed = time() - start_time
    print(f"Elapsed wall time {timedelta(seconds=int(elapsed))}")


def build_sequential(builds, build_dir, args):
    completed = []

    for bd in builds:
        print(f"Building {', '.join(bd['tags'])}...", end="")

        cmd_string = "$ "+" ".join(bd['cmd'])

        log_file = path.join(args.log_dir, bd['tags'][0]+'.log')
        with open(log_file, 'w') as f:
            f.write(cmd_string)
            f.write("\n\n")

        if args.verbose or args.dry_run:
            print()
            print(cmd_string)
            print()

        if not args.dry_run:
            build_output = ""
            prev_time = 0

            with Popen(bd['cmd'],
                       cwd=build_dir,
                       stdout=PIPE,
                       bufsize=1,
                       text=True) as sp:
                for line in sp.stdout:
                    with open(log_file, 'a') as f:
                        f.write(line)

                    if args.verbose:
                        print(line, end="")
                    else:
                        build_output += line
                        if time() - prev_time > 5:
                            print('.', end="", flush=True)
                            prev_time = time()

            print()

            if sp.returncode != 0:
                print("Build failed!")
                if not args.verbose:
                    print("Output:")
                    print()
                    print(build_output)
                sys.exit()

            else:
                completed.append(bd['tags'])

    if completed:
        print("Successfully completed builds",
              ", ".join("("+", ".join(tags)+")" for tags in completed))
        print()


def build_parallel(builds, build_dir, args):

    with ExitStack() as stack:

        for bd in builds:
            print(f"Building {', '.join(bd['tags'])}...")

            bd['log_file'] = path.join(args.log_dir, bd['tags'][0]+'.log')
            with open(bd['log_file'], 'w') as f:
                cmd_string = "$ "+" ".join(bd['cmd'])
                f.write(cmd_string)
                f.write("\n\n")

            bd['proc'] = Popen(bd['cmd'],
                               cwd=build_dir,
                               stdout=PIPE,
                               stderr=PIPE,
                               bufsize=1,
                               text=True)

            # allow non-blocking output reads
            set_blocking(bd['proc'].stdout.fileno(), False)
            set_blocking(bd['proc'].stderr.fileno(), False)

            bd['completed'] = False

            stack.enter_context(bd['proc'])
            sleep(0.5)

        while not all(bd['completed'] for bd in builds):
            sleep(1)

            new_output = None
            for bd in builds:
                returncode = bd['proc'].poll()
                if not bd['completed'] and returncode is not None:
                    bd['completed'] = True
                    if returncode == 0:
                        print()
                        print("Successfully completed "
                              f"{', '.join(bd['tags'])}")
                    else:
                        print()
                        print(f"Build failed for {', '.join(bd['tags'])}")
                        print(f"See {bd['log_file']} for details")

                    new_output = False

                for line in combine_stdout_stderr(bd['proc']):
                    if new_output is None:
                        new_output = True

                    with open(bd['log_file'], 'a') as f:
                        f.write(line)

            if new_output is True:
                print('.', end="", flush=True)

    print()


def combine_stdout_stderr(p):
    for line in p.stdout:
        if not line:  # we've gone through all the new input
            break
        yield line

    for line in p.stderr:
        if not line:
            break
        yield line


if __name__ == '__main__':
    main()
