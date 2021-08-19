#! /usr/bin/env python

from pathos import multiprocessing
from itertools import product
import numpy as np
import traceback
import os

import sys

sys.path.append("..")

from lib.problems import get_problem

TM_DIR = "../traffic-matrices"
SCALE_FACTORS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
MODELS = ["gravity", "uniform", "poisson-high-intra", "poisson-high-inter", "bimodal"]
NUM_SAMPLES = 5


def generate_traffic_matrix(args):
    prob_short_name, model, scale_factor = args
    tm_model_dir = os.path.join(TM_DIR, model)

    for _ in range(NUM_SAMPLES):
        print(prob_short_name, model, scale_factor)
        problem = get_problem(
            prob_short_name,
            model,
            scale_factor=scale_factor,
            seed=np.random.randint(2 ** 31 - 1),
        )
        problem.print_stats()

        try:
            problem.traffic_matrix.serialize(tm_model_dir)
        except Exception:
            print(
                "{}, model {}, scale factor {} failed".format(
                    problem.name, model, scale_factor
                )
            )
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
        "erdos-renyi-1260231677",
    ]
    if len(sys.argv) == 2 and sys.argv[1] == "--holdout":
        TM_DIR += "/holdout"

    if not os.path.exists(TM_DIR):
        os.makedirs(TM_DIR)
    for model in MODELS:
        tm_model_dir = os.path.join(TM_DIR, model)
        if not os.path.exists(tm_model_dir):
            os.makedirs(tm_model_dir)

    pool = multiprocessing.ProcessPool(14)
    pool.map(
        generate_traffic_matrix, product(PROBLEM_SHORT_NAMES, MODELS, SCALE_FACTORS)
    )
