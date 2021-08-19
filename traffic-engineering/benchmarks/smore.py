#! /usr/bin/env python

from benchmark_consts import get_args_and_problems, print_

import os
import pickle
import traceback
import numpy as np

import sys

sys.path.append("..")

from lib.algorithms import SMORE
from lib.problem import Problem

TOP_DIR = "smore-logs"
HEADERS = [
    "problem",
    "num_nodes",
    "num_edges",
    "traffic_seed",
    "tm_model",
    "scale_factor",
    "num_commodities",
    "total_demand",
    "algo",
    "num_paths",
    "total_flow",
    "runtime",
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)


def benchmark(problems):
    with open("smore.csv", "a") as w:
        print_(",".join(HEADERS), file=w)
        for problem_name, topo_fname, tm_fname in problems:

            problem = Problem.from_file(topo_fname, tm_fname)
            print_(problem.name, tm_fname)

            traffic_seed = problem.traffic_matrix.seed
            total_demand = np.sum(problem.traffic_matrix.tm)
            print_("traffic seed: {}".format(traffic_seed))
            print_("traffic matrix model: {}".format(problem.traffic_matrix.model))
            print_("total demand: {}".format(total_demand))

            run_dir = os.path.join(
                TOP_DIR,
                problem.name,
                "{}-{}".format(traffic_seed, problem.traffic_matrix.model),
            )
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

            try:
                with open(
                    os.path.join(run_dir, "{}-smore.txt".format(problem.name)), "w"
                ) as log:
                    smore = SMORE.new_total_flow(num_paths=4, out=log)
                    smore.solve(problem)
                    smore_sol_dict = smore.sol_dict
                    pickle.dump(
                        smore_sol_dict,
                        open(
                            os.path.join(
                                run_dir, "{}-smore-sol-dict.pkl".format(problem.name)
                            ),
                            "wb",
                        ),
                    )

                    result_line = PLACEHOLDER.format(
                        problem.name,
                        len(problem.G.nodes),
                        len(problem.G.edges),
                        traffic_seed,
                        problem.traffic_matrix.model,
                        problem.traffic_matrix.scale_factor,
                        len(problem.commodity_list),
                        total_demand,
                        "smore",
                        4,
                        smore.total_flow,
                        smore.runtime,
                    )
                    print_(result_line, file=w)

            except Exception:
                print_(
                    "SMORE, Problem {}, traffic seed {}, traffic model {} failed".format(
                        problem.name, traffic_seed, problem.traffic_matrix.model
                    )
                )
                traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, problems = get_args_and_problems()

    if args.dry_run:
        print("Problems to run:")
        for problem in problems:
            print(problem)
    else:
        benchmark(problems)
