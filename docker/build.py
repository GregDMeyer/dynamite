"""
This script builds Docker images for various combinations of
parameters. Should be run from inside the git tree.
"""

import sys
from os import path, mkdir, set_blocking
from argparse import ArgumentParser
from subprocess import run, PIPE, Popen
from contextlib import ExitStack
from time import time, sleep
from datetime import timedelta
import shutil


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser(argv)

    parser.add_argument("--targets", type=lambda x: x.split(','),
                        default=['release', 'jupyter'],
                        help='Targets to build, delimited by commas.')

    parser.add_argument("--hardware", type=lambda x: x.split(','),
                        default=['cpu', 'gpu'],
                        help='Whether to build the CPU and/or GPU versions.')

    parser.add_argument("-v", "--verbose", action='store_true',
                        help='Show build output.')

    parser.add_argument("--fresh", action='store_true',
                        help='Update cached images.')

    parser.add_argument("--cuda-archs", type=lambda x: x.split(','),
                        default=['70', '80'],
                        help='CUDA compute capabilities.')

    parser.add_argument("--dry-run", action="store_true",
                        help="Print build commands but do not run them.")

    parser.add_argument("--save-logs", action="store_true",
                        help="Save build output to /tmp/dnm_build.")

    parser.add_argument("-p", "--parallel", action="store_true",
                        help="Build independent images in parallel.")

    args = parser.parse_args()

    if args.verbose and args.parallel:
        raise ValueError("Cannot run verbose build in parallel.")

    if args.dry_run:
        args.parallel = False

    # I don't know if we actually need a flag for this
    args.log_dir = "/tmp/dnm_build_logs"

    return args


def main():
    args = parse_args()

    # checkout a clean version to build from
    git_root = run(["git", "rev-parse", "--show-toplevel"],
                   capture_output=True, text=True, check=True).stdout.strip()

    build_dir = "/tmp/dnm_docker_build"
    if path.exists(build_dir):
        build_dir += '_'+str(int(time()))

    if not path.exists(args.log_dir):
        mkdir(args.log_dir)

    run(["git", "clone", git_root, build_dir], check=True)

    version = open(path.join(build_dir, 'VERSION')).read().strip()

    start_time = time()

    # run builds
    first_target = True
    for target in args.targets:
        builds = []
        for hardware in args.hardware:
            if hardware == 'gpu':
                cuda_archs = args.cuda_archs
            else:
                cuda_archs = [None]

            for cuda_arch in cuda_archs:

                tags = ["latest", version]

                if hardware == 'gpu':
                    tags = [tag+'-cuda' for tag in tags]
                    no_cc_tags = tags.copy()
                    tags = [tag+'.cc'+cuda_arch for tag in tags]

                    # default is cc 7.0
                    if cuda_arch == '70':
                        tags = no_cc_tags + tags

                if target == 'jupyter':
                    tags = [tag+'-jupyter' for tag in tags]

                cmd = [
                    "docker", "build",
                    "--build-arg", f"HARDWARE={hardware}",
                    "-f", "docker/Dockerfile",
                    "--target", target
                ]

                if cuda_arch is not None:
                    cmd += ["--build-arg", f"CUDA_ARCH={cuda_arch}"]

                if args.fresh and first_target:
                    cmd += ["--no-cache", "--pull"]

                for tag in tags:
                    cmd += ["-t", f"gdmeyer/dynamite:{tag}"]

                cmd += ["."]

                build_dict = {}
                build_dict['cmd'] = cmd
                build_dict['tags'] = tags

                builds.append(build_dict)

        if args.parallel:
            build_parallel(builds, build_dir, args)
        else:
            build_sequential(builds, build_dir, args)

        first_target = False

    elapsed = time() - start_time
    print(f"Builds completed in {timedelta(seconds=int(elapsed))}")

    print("Removing build files...")
    if not build_dir.startswith("/tmp"):
        # something has gone horribly wrong
        print("not removing build files, not in /tmp")
    else:
        shutil.rmtree(build_dir)


def build_sequential(builds, build_dir, args):
    completed = []

    for bd in builds:
        print(f"Building {', '.join(bd['tags'])}...", end="")

        cmd_string = "$ "+" ".join(bd['cmd'])

        if args.save_logs:
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

            with Popen(bd['cmd'], cwd=build_dir, stdout=PIPE, bufsize=1, text=True) as sp:
                for line in sp.stdout:
                    if args.save_logs:
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
        print("Successfully completed builds", ", ".join("("+", ".join(tags)+")" for tags in completed))
        print()


def build_parallel(builds, build_dir, args):

    with ExitStack() as stack:

        for bd in builds:
            print(f"Building {', '.join(bd['tags'])}...")

            if args.save_logs:
                bd['log_file'] = path.join(args.log_dir, bd['tags'][0]+'.log')
                with open(bd['log_file'], 'w') as f:
                    cmd_string = "$ "+" ".join(bd['cmd'])
                    f.write(cmd_string)
                    f.write("\n\n")

            bd['proc'] = Popen(bd['cmd'],
                               cwd=build_dir,
                               stdout=PIPE,
                               bufsize=1,
                               text=True)

            # allow non-blocking stdout reads
            set_blocking(bd['proc'].stdout.fileno(), False)

            bd['completed'] = False

            stack.enter_context(bd['proc'])

        while not all(bd['completed'] for bd in builds):
            sleep(1)

            new_output = None
            for bd in builds:
                returncode = bd['proc'].poll()
                if not bd['completed'] and returncode is not None:
                    bd['completed'] = True
                    if returncode == 0:
                        print()
                        print(f"Successfully completed {', '.join(bd['tags'])}")
                    else:
                        print()
                        print(f"Build failed for {', '.join(bd['tags'])}")
                        print(f"See {bd['log_file']} for details")

                    new_output = False

                for line in bd['proc'].stdout:
                    if not line:  # we've gone through all the new input
                        break

                    if new_output is None:
                        new_output = True

                    if args.save_logs:
                        with open(bd['log_file'], 'a') as f:
                            f.write(line)

            if new_output is True:
                print('.', end="", flush=True)

    print()


if __name__ == '__main__':
    main()
