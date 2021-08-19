#! /usr/bin/env python

# Use this script to generate a full TM so we can
# compute the fib entries for a given topology
import sys

sys.path.append("..")

from lib.problems import get_problem
from pathos import multiprocessing
import os

TM_DIR = "../traffic-matrices/full-tms"


def generate_traffic_matrix(prob_short_name):
    problem = get_problem(prob_short_name, "gravity", scale=1.0, random=False)
    assert problem.traffic_matrix.is_full
    print(problem.name)
    problem.print_stats()

    try:
        problem.traffic_matrix.serialize(TM_DIR)
    except Exception:
        print("{} failed".format(problem.name))
        import traceback

        traceback.printexc()


if __name__ == "__main__":
    PROBLEM_SHORT_NAMES = [
        "gtsce",
        "delta",
        "us-carrier",
        "tata",
        "cogentco",
        "dial",
        "colt",
        "interoute",
        "ion",
        "uninett",
        "kdl",
    ]
    if not os.path.exists(TM_DIR):
        os.makedirs(TM_DIR)

    pool = multiprocessing.ProcessPool(len(PROBLEM_SHORT_NAMES))
    pool.map(generate_traffic_matrix, PROBLEM_SHORT_NAMES)
