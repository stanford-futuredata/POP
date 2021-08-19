#! /usr/bin/env python

import os
import subprocess
from glob import iglob
from pathos import multiprocessing

import sys

sys.path.append("..")

from lib.config import TL_DIR
from benchmarks.benchmark_consts import PROBLEM_NAMES

YATES_HOME_DIR = os.path.join(os.getenv("HOME"), "src", "yates")
YATES_TOPO_DIR = os.path.abspath(os.path.join(TL_DIR, "topologies", "yates-format"))
YATES_TM_DIR = os.path.abspath(os.path.join(TL_DIR, "traffic-matrices", "yates-format"))
YATES_HOSTS_DIR = os.path.join(YATES_TM_DIR, "hosts")
YATES_RESULTS_DIR = os.path.join("data", "results")
OUTPUT_DIR = os.path.join(TL_DIR, "topologies", "paths", "raeke")

# yates ${TL_DIR}/topologies/yates-format/GtsCe.dot \
#   ${TL_DIR}/traffic-matrices/yates-format/GtsCe.graphml_traffic-matrix.txt \
#   ${TL_DIR}/traffic-matrices/yates-format/GtsCe.graphml_traffic-matrix.txt \
#   ${TL_DIR}/topologies/yates-format/GtsCe.hosts -raeke -budget 4
def run_yates(args):
    os.chdir(YATES_HOME_DIR)
    problem_name, tm_fname, num_paths = args
    problem_name_dot = problem_name.replace(".graphml", ".dot").replace(".json", ".dot")
    cmd = [
        "yates",
        os.path.join(YATES_TOPO_DIR, problem_name_dot),
        os.path.join(YATES_TM_DIR, tm_fname),
        os.path.join(YATES_TM_DIR, tm_fname),
        os.path.join(YATES_HOSTS_DIR, problem_name_dot.replace(".dot", ".hosts")),
        "-raeke",
        "-budget",
        str(num_paths),
    ]
    print(" ".join(cmd))
    print()
    subprocess.call(cmd)

    os.chdir(
        os.path.join(YATES_RESULTS_DIR, problem_name_dot.replace(".dot", ""), "paths")
    )
    os.rename(
        "raeke_0",
        os.path.join(OUTPUT_DIR, "{}-{}-paths-rrt.txt".format(problem_name, num_paths)),
    )


# NOTE: run eval `opam config env` before executing
if __name__ == "__main__":
    num_paths = int(sys.argv[1])
    run_args = []
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for problem_name in PROBLEM_NAMES:
        print(problem_name)
        for tm_fname in iglob(
            os.path.join(YATES_TM_DIR, "{}_traffic-matrix.txt".format(problem_name))
        ):
            run_args.append((problem_name, tm_fname, num_paths))

    pool = multiprocessing.ProcessPool(7)
    pool.map(run_yates, run_args)
