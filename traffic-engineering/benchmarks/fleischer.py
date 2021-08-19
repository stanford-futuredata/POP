#! /usr/bin/env python

import os
import subprocess
from glob import iglob
from pathos import multiprocessing

import sys

sys.path.append("..")

from lib.config import TL_DIR

INPUT_DIR_PATHS = os.path.abspath(
    os.path.join(TL_DIR, "traffic-matrices", "fleischer-with-paths-format")
)
INPUT_DIR_EDGE = os.path.abspath(
    os.path.join(TL_DIR, "traffic-matrices", "fleischer-edge-format")
)
OUTPUT_DIR_PATHS = os.path.abspath(
    os.path.join(TL_DIR, "benchmarks", "fleischer-runs", "with-paths")
)
OUTPUT_DIR_EDGE = os.path.abspath(
    os.path.join(TL_DIR, "benchmarks", "fleischer-runs", "edge")
)
FLEISCHER_RUN_DIR = os.path.abspath(os.path.join(TL_DIR, "ext", "fleischer"))


def run_fleischer(args):
    input_fname, epsilon, paths, output_fname = args
    if paths:
        flag = "-22p"
    else:
        flag = "-f"
    cmd = ["./fl", flag, input_fname, str(epsilon)]
    print(cmd + [output_fname])

    with open(output_fname, "w") as w:
        subprocess.call(cmd, stdout=w)


if __name__ == "__main__":
    if sys.argv[1] == "path":
        paths = True
    elif sys.argv[1] == "edge":
        paths = False
    else:
        raise Exception("invalid arg {}".format(sys.argv[1]))

    if sys.argv[2] != "--slice":
        raise Exception("missing --slice")
    slice = int(sys.argv[3])

    dry_run = sys.argv[-1] == "--dry-run"

    if paths:
        input_dir = os.path.join(INPUT_DIR_PATHS, "slice-{}".format(slice))
        output_dir = os.path.join(OUTPUT_DIR_PATHS, "slice-{}".format(slice))
    else:
        input_dir = os.path.join(INPUT_DIR_EDGE, "slice-{}".format(slice))
        output_dir = os.path.join(OUTPUT_DIR_EDGE, "slice-{}".format(slice))

    if not os.path.exists(input_dir):
        print(
            "{} does not exist; cannot run benchmark without input files. Run scripts/serialize_all_fleischer.py".format(
                input_dir
            )
        )
        exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(FLEISCHER_RUN_DIR)
    subprocess.call(["make"])

    run_args = []
    for input_fname in iglob(input_dir + "/*"):
        for epsilon in [0.5]:
            output_fname = os.path.join(
                output_dir,
                os.path.basename(input_fname).replace(
                    ".txt", "_epsilon-{}.output".format(epsilon)
                ),
            )
            if os.path.exists(output_fname):
                continue
            run_args.append((input_fname, epsilon, paths, output_fname))
    if dry_run:
        print("Problems to run:")
        for run_arg in run_args:
            print(run_arg)
    else:
        # Only run 4 jobs at once
        pool = multiprocessing.ProcessPool(4)
        pool.map(run_fleischer, run_args)
