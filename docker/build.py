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

    parser.add_argument("--fresh", action='store_true',
                        help='Update cached images.')

    return parser.parse_args()


def main():
    args = parse_args()

    # checkout a clean version to build from
    git_root = run(["git", "rev-parse", "--show-toplevel"],
                   capture_output=True, text=True, check=True).stdout.strip()

    build_dir = "/tmp/dnm_docker_build"
    if path.exists(build_dir):
        build_dir += '_'+str(int(time()))

    run(["git", "clone", git_root, build_dir], check=True)

    version = open(path.join(build_dir, 'VERSION')).read().strip()

    completed = []

    # run builds
    for target in args.targets:
        for hardware in args.hardware:
            tags = ["latest", version]

            tag_append = ""
            if hardware == 'gpu':
                tag_append += '-cuda'

            if target == 'jupyter':
                tag_append += '-jupyter'

            cmd = [
                "docker", "build",
                "--build-arg", f"HARDWARE={hardware}",
                "-f", "docker/Dockerfile",
                "--target", target
            ]

            if args.fresh and (target == 'release' or 'release' not in args.targets):
                cmd += ["--no-cache", "--pull"]

            this_build_tags = []
            for tag_base in tags:
                tag = tag_base+tag_append
                cmd += ["-t", f"gdmeyer/dynamite:{tag}"]
                this_build_tags.append(tag)

            cmd += ["."]

            print(f"Building {', '.join(this_build_tags)}...", end="")

            if args.verbose:
                print()
                print()
                print("$ "+" ".join(cmd))

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
                completed += this_build_tags

    print("Removing build files...")
    if not build_dir.startswith("/tmp"):
        # something has gone horribly wrong
        print("not removing build files, not in /tmp")
    else:
        shutil.rmtree(build_dir)

    print("Successfully completed builds", ", ".join(completed))


if __name__ == '__main__':
    main()
