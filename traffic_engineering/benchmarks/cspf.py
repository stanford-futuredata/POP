#! /usr/bin/env python

from benchmark_helpers import get_args_and_problems, print_, PATH_FORM_HYPERPARAMS

import os

# import pickle
import traceback

import sys

sys.path.append("..")

from lib.algorithms import CSPF, Objective
from lib.problem import Problem

TOP_DIR = "cspf-logs"
HEADERS = [
    "problem",
    "num_nodes",
    "num_edges",
    "traffic_seed",
    "scale_factor",
    "tm_model",
    "num_commodities",
    "total_demand",
    "algo",
    "num_paths",
    "edge_disjoint",
    "dist_metric",
    "objective",
    "obj_val",
    "runtime",
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)

OUTPUT_CSV_TEMPLATE = "cspf-{}-{}.csv"

# Sweep topos and traffic matrices for that topo. For each combo, record the
# runtime and total flow for each algorithm
def benchmark(problems, output_csv, obj):
    num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
    with open(output_csv, "a") as results:
        print_(",".join(HEADERS), file=results)
        for problem_name, topo_fname, tm_fname in problems:
            problem = Problem.from_file(topo_fname, tm_fname)
            print_(problem_name, tm_fname)
            traffic_seed = problem.traffic_matrix.seed
            total_demand = problem.total_demand
            print_("traffic seed: {}".format(traffic_seed))
            print_(
                "traffic scale factor: {}".format(problem.traffic_matrix.scale_factor)
            )
            print_("traffic matrix model: {}".format(problem.traffic_matrix.model))
            print_("total demand: {}".format(total_demand))

            run_dir = os.path.join(
                TOP_DIR,
                problem_name,
                "{}-{}".format(traffic_seed, problem.traffic_matrix.model),
            )
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

            try:
                print_(
                    "\nCSPF, objective {}, {} paths, edge disjoint {}, dist metric {}".format(
                        obj, num_paths, edge_disjoint, dist_metric
                    )
                )
                with open(
                    os.path.join(
                        run_dir,
                        "{}-cspf_objective-{}_{}-paths_edge-disjoint-{}_dist-metric-{}.txt".format(
                            problem_name, obj, num_paths, edge_disjoint, dist_metric
                        ),
                    ),
                    "w",
                ) as log:
                    cspf = CSPF(
                        objective=Objective.get_obj_from_str(obj),
                        num_paths=num_paths,
                        edge_disjoint=edge_disjoint,
                        dist_metric=dist_metric,
                        out=log,
                    )
                    cspf.solve(problem)
                    # TODO: add this in later
                    # cspf_sol_dict = cspf.sol_dict
                    # with open(
                    #     os.path.join(
                    #         run_dir,
                    #         "{}-cspf_objective-{}_{}-paths_edge-disjoint-{}_dist-metric-{}_sol-dict.pkl".format(
                    #             problem_name, obj, num_paths, edge_disjoint, dist_metric
                    #         ),
                    #     ),
                    #     "wb",
                    # ) as w:
                    #     pickle.dump(cspf_sol_dict, w)

                result_line = PLACEHOLDER.format(
                    problem_name,
                    len(problem.G.nodes),
                    len(problem.G.edges),
                    traffic_seed,
                    problem.traffic_matrix.scale_factor,
                    problem.traffic_matrix.model,
                    len(problem.commodity_list),
                    total_demand,
                    "cspf",
                    num_paths,
                    edge_disjoint,
                    dist_metric,
                    obj,
                    cspf.obj_val,
                    cspf.runtime,
                )
                print_(result_line, file=results)

            except Exception:
                print_(
                    "CSPF, objective {}, {} paths, edge disjoint {}, dist metric {}, Problem {}, traffic seed {}, traffic model {} failed".format(
                        obj,
                        num_paths,
                        edge_disjoint,
                        dist_metric,
                        problem_name,
                        traffic_seed,
                        problem.traffic_matrix.model,
                    )
                )
                traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, output_csv, problems = get_args_and_problems(OUTPUT_CSV_TEMPLATE)
    # problems = problems[294:295] # Kdl.graphml, traffic_seed = 1710674203, gravity, 16.0

    if args.dry_run:
        print("Problems to run:")
        for problem in problems:
            print(problem)
    else:
        benchmark(problems, output_csv, args.obj)
