"""
This script builds Docker images for various combinations of
parameters. Should be run from inside the git tree.
"""

import sys
from os import path
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

    return parser.parse_args()


def main():
    args = parse_args()

    git_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                     capture_output=True, text=True, check=True).stdout.strip()
    git_commit = run(["git", "describe", "--always"],
                     capture_output=True, text=True, check=True).stdout.strip()

    # checkout a clean version to build from
    git_root = run(["git", "rev-parse", "--show-toplevel"],
                   capture_output=True, text=True, check=True).stdout.strip()

    build_dir = "/tmp/dnm_docker_build"
    if path.exists(build_dir):
        build_dir += '_'+str(int(time()))

    run(["git", "clone", git_root, build_dir], check=True)

    completed = []

    # run builds
    for target in args.targets:
        for hardware in args.hardware:

            dockerfile = f"Dockerfile-{hardware}"
            tag = "latest"

            if hardware == 'gpu':
                tag += '-cuda'

            if target == 'jupyter':
                tag += '-jupyter'

            cmd = [
                "docker", "build",
                "--build-arg", f"GIT_BRANCH={git_branch}",
                "--build-arg", f"GIT_COMMIT={git_commit}",
                "-f", f"docker/{dockerfile}",
                "--target", target,
                "-t", f"gdmeyer/dynamite:{tag}",
                "."
            ]

            print(f"Building '{tag}'...", end="")

            if args.verbose:
                print()
                print()

            build_output = ""
            prev_time = 0

            with Popen(cmd, cwd=build_dir, stdout=PIPE, bufsize=1, text=True) as sp:
                for line in sp.stdout:
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
                completed.append(tag)

    print("Removing build files...")
    if not build_dir.startswith("/tmp"):
        # something has gone horribly wrong
        print("not removing build files, not in /tmp")
    else:
        shutil.rmtree(build_dir)

    print("Successfully completed builds", ", ".join(completed))


if __name__ == '__main__':
    main()
