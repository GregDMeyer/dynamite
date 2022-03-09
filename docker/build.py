"""
This script builds Docker images for various combinations of
parameters. Should be run from inside the git tree.
"""

import sys
from os import path, mkdir
from argparse import ArgumentParser
from subprocess import run, PIPE, Popen
from time import time
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

    return parser.parse_args()


def main():
    args = parse_args()

    # checkout a clean version to build from
    git_root = run(["git", "rev-parse", "--show-toplevel"],
                   capture_output=True, text=True, check=True).stdout.strip()

    build_dir = "/tmp/dnm_docker_build"
    if path.exists(build_dir):
        build_dir += '_'+str(int(time()))

    log_dir = "/tmp/dnm_build"
    if not path.exists(log_dir):
        mkdir(log_dir)

    run(["git", "clone", git_root, build_dir], check=True)

    version = open(path.join(build_dir, 'VERSION')).read().strip()

    completed = []

    # run builds
    for target in args.targets:
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
                        tags += no_cc_tags

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

                if args.fresh and (target == 'release' or 'release' not in args.targets):
                    cmd += ["--no-cache", "--pull"]

                for tag in tags:
                    cmd += ["-t", f"gdmeyer/dynamite:{tag}"]

                cmd += ["."]

                print(f"Building {', '.join(tags)}...", end="")

                cmd_string = "$ "+" ".join(cmd)

                if args.save_logs:
                    log_file = path.join(log_dir, tags[0]+'.log')
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

                    with Popen(cmd, cwd=build_dir, stdout=PIPE, bufsize=1, text=True) as sp:
                        for line in sp.stdout:
                            if args.save_logs:
                                with open(log_file, 'a') as f:
                                    f.write(line)
                            if args.verbose:
                                print(line, end="")
                            else:
                                build_output += line
                                if time() - prev_time > 1:
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
                        completed += tags

    print("Removing build files...")
    if not build_dir.startswith("/tmp"):
        # something has gone horribly wrong
        print("not removing build files, not in /tmp")
    else:
        shutil.rmtree(build_dir)

    if completed:
        print("Successfully completed builds", ", ".join(completed))


if __name__ == '__main__':
    main()
