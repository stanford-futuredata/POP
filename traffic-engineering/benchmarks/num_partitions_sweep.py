#! /usr/bin/env python

from benchmark_consts import get_args_and_problems, print_, NCFLOW_HYPERPARAMS
from fib_entries import get_fib_entry_problem

import os
import traceback
import numpy as np

import sys

sys.path.append("..")

from lib.algorithms import NcfEpi
from lib.problem import Problem
from lib.partitioning.utils import all_partitions_contiguous

TOP_DIR = "num-parts-sweep-logs"
TOPOLOGY = "Kdl.graphml"
TM_MODEL = "uniform"
SCALE_FACTOR = 32.0

# Sweep topos and traffic matrices for that topo. For each combo, record the
# runtime and total flow for each algorithm
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
    "clustering_algo",
    "num_partitions",
    "size_of_largest_partition",
    "partition_runtime",
    "num_paths",
    "edge_disjoint",
    "dist_metric",
    "total_flow",
    "runtime",
    "num_iters",
    "total_num_fib_entries",
    "max_num_fib_entries",
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)


def benchmark(problems, scale_factors_to_sweep=[0.25, 0.5, 1, 2, 3, 4, 5, 6, 7]):

    with open("num-partitions-sweep.csv", "a") as results:
        print_(",".join(HEADERS), file=results)
        for problem_name, topo_fname, tm_fname in problems:
            problem = Problem.from_file(topo_fname, tm_fname)
            print_(problem.name, tm_fname)
            traffic_seed = problem.traffic_matrix.seed
            total_demand = problem.total_demand
            print_("traffic seed: {}".format(traffic_seed))
            print_("traffic matrix model: {}".format(problem.traffic_matrix.model))
            print_(
                "traffic matrix scale factor: {}".format(
                    problem.traffic_matrix.scale_factor
                )
            )
            print_("total demand: {}".format(total_demand))

            (
                num_paths,
                edge_disjoint,
                dist_metric,
                partition_cls,
                _,
            ) = NCFLOW_HYPERPARAMS[problem_name]
            for scale_factor in scale_factors_to_sweep:
                num_partitions_to_set = int(
                    scale_factor * int(np.sqrt(len(problem.G.nodes)))
                )
                print(
                    "Scale factor: {} -> num partitions: {}".format(
                        scale_factor, num_partitions_to_set
                    )
                )
                partitioner = partition_cls(num_partitions_to_set)
                partition_algo = partitioner.name

                partition_vector = partitioner.partition(problem)
                num_parts = len(np.unique(partition_vector))
                if not all_partitions_contiguous(problem, partition_vector):
                    print_(
                        "Problem {}, partitioner {}, num_partitions_to_set {} did not find a valid partition".format(
                            problem_name, partition_algo, num_parts
                        )
                    )
                    continue

                run_dir = os.path.join(
                    TOP_DIR,
                    problem.name,
                    "{}-{}".format(traffic_seed, problem.traffic_matrix.model),
                )
                if not os.path.exists(run_dir):
                    os.makedirs(run_dir)

                try:
                    print_(
                        "\nNCFlow, {} partitioner, {} partitions, {} paths, edge disjoint {}, dist metric {}".format(
                            partition_algo,
                            num_partitions_to_set,
                            num_paths,
                            edge_disjoint,
                            dist_metric,
                        )
                    )
                    run_nc_dir = os.path.join(
                        run_dir,
                        "ncflow",
                        partition_algo,
                        "{}-partitions".format(num_partitions_to_set),
                        "{}-paths".format(num_paths),
                        "edge_disjoint-{}".format(edge_disjoint),
                        "dist_metric-{}".format(dist_metric),
                    )
                    if not os.path.exists(run_nc_dir):
                        os.makedirs(run_nc_dir)

                    fib_entry_problem = get_fib_entry_problem(problem.name)
                    print_(
                        "NCFlowEdgePerIter, Problem {}, partitioner {}, {} partitions, {} paths, edge disjoint {}, dist_metric {} computing fib entries".format(
                            fib_entry_problem.name,
                            partition_algo,
                            num_parts,
                            num_paths,
                            edge_disjoint,
                            dist_metric,
                        )
                    )

                    total_num_fib_entries, max_num_fib_entries = NcfEpi.fib_entries(
                        fib_entry_problem,
                        partitioner,
                        num_paths,
                        edge_disjoint,
                        dist_metric,
                    )
                    print_(
                        "NCFlowEdgePerIter, Problem {}, partitioner {}, {} partitions, {} paths, edge disjoint {}, dist_metric {}, total num fib entries: {}".format(
                            problem_name,
                            partition_algo,
                            num_parts,
                            num_paths,
                            edge_disjoint,
                            dist_metric,
                            total_num_fib_entries,
                        )
                    )
                    print_(
                        "NCFlowEdgePerIter, Problem {}, partitioner {}, {} partitions, {} paths, edge disjoint {}, dist_metric {}, max num fib entries: {}".format(
                            problem_name,
                            partition_algo,
                            num_parts,
                            num_paths,
                            edge_disjoint,
                            dist_metric,
                            max_num_fib_entries,
                        )
                    )

                    with open(
                        os.path.join(
                            run_nc_dir,
                            "{}-ncflow-partitioner_{}-{}_partitions-{}_paths-edge_disjoint_{}-dist_metric_{}.txt".format(
                                problem.name,
                                partition_algo,
                                num_partitions_to_set,
                                num_paths,
                                edge_disjoint,
                                dist_metric,
                            ),
                        ),
                        "w",
                    ) as log:
                        ncflow = NcfEpi.new_total_flow(
                            num_paths,
                            edge_disjoint=edge_disjoint,
                            dist_metric=dist_metric,
                            out=log,
                        )
                        ncflow.solve(problem, partitioner)

                        num_partitions = len(np.unique(ncflow._partition_vector))
                        runtime = ncflow.runtime_est(14)
                        total_flow = ncflow.obj_val

                        result_line = PLACEHOLDER.format(
                            problem.name,
                            len(problem.G.nodes),
                            len(problem.G.edges),
                            traffic_seed,
                            problem.traffic_matrix.model,
                            problem.traffic_matrix.scale_factor,
                            len(problem.commodity_list),
                            total_demand,
                            "ncflow_edge_per_iter",
                            partition_algo,
                            num_partitions,
                            partitioner.size_of_largest_partition,
                            partitioner.runtime,
                            num_paths,
                            edge_disjoint,
                            dist_metric,
                            total_flow,
                            runtime,
                            ncflow.num_iters,
                            total_num_fib_entries,
                            max_num_fib_entries,
                        )
                        print_(result_line, file=results)
                except:
                    print_(
                        "NCFlowEdgePerIter partitioner {}, {} paths, Problem {}, traffic seed {}, traffic model {} failed".format(
                            partition_algo,
                            num_paths,
                            problem.name,
                            traffic_seed,
                            problem.traffic_matrix.model,
                        )
                    )
                    traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, problems = get_args_and_problems()
    problems = [
        problem
        for problem in problems
        if problem[0] == TOPOLOGY
        and TM_MODEL in problem[-1]
        and "_{}_".format(SCALE_FACTOR) in problem[-1]
    ]

    if args.dry_run:
        print("Problems to run:")
        for problem in problems:
            print(problem)
    else:
        benchmark(problems, scale_factors_to_sweep=[0.25, 0.5, 1, 2, 3, 4, 5, 6, 7])
