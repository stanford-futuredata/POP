#! /usr/bin/env python

from collections import defaultdict
from glob import iglob
import numpy as np
import os

import sys

sys.path.append("..")

from lib.problem import Problem
from lib.partitioning.utils import all_partitions_contiguous
from lib.algorithms import NcfEpi, PathFormulation, SMORE
from benchmark_consts import (
    PROBLEM_NAMES,
    NCFLOW_HYPERPARAMS,
    PATH_FORM_HYPERPARAMS,
    print_,
)

# Sweep topos and traffic matrices for that topo. For each combo, record the
# runtime and total flow for each algorithm
HEADERS = [
    "problem",
    "num_nodes",
    "num_edges",
    "traffic_seed",
    "tm_model",
    "num_commodities",
    "total_demand",
    "algo",
    "clustering_algo",
    "num_partitions",
    "partition_runtime",
    "num_paths",
    "edge_disjoint",
    "dist_metric",
    "total_num_fib_entries",
    "max_num_fib_entries",
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)


def get_fib_entry_problem(problem_name):
    if problem_name.endswith(".graphml"):
        topo_fname = os.path.join("..", "topologies", "topology-zoo", problem_name)
    else:
        topo_fname = os.path.join("..", "topologies", problem_name)

    for tm_fname in iglob(
        "../traffic-matrices/full-tms/{}*_traffic-matrix.pkl".format(problem_name)
    ):
        return Problem.from_file(topo_fname, tm_fname)


def benchmark(
    problem_names,
    benchmark_nc=True,
    benchmark_path=True,
    benchmark_smore=True,
    benchmark_fleischer_edge=True,
):

    with open("fib-entries.csv", "a") as results:
        print_(",".join(HEADERS), file=results)

        for problem_name in problem_names:
            if benchmark_path:
                num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
                problem = get_fib_entry_problem(problem_name)
                (
                    total_num_fib_entries,
                    max_num_fib_entries,
                ) = PathFormulation.fib_entries(
                    problem, num_paths, edge_disjoint, dist_metric
                )
                print_(
                    "Path Formulation, Problem {}, {} paths, edge disjoint {}, dist_metric {}, total num fib entries: {}".format(
                        problem_name,
                        num_paths,
                        edge_disjoint,
                        dist_metric,
                        total_num_fib_entries,
                    )
                )
                print_(
                    "Path Formulation, Problem {}, {} paths, edge disjoint {}, dist_metric {}, max num fib entries: {}".format(
                        problem_name,
                        num_paths,
                        edge_disjoint,
                        dist_metric,
                        max_num_fib_entries,
                    )
                )

                result_line = PLACEHOLDER.format(
                    problem_name,
                    len(problem.G.nodes),
                    len(problem.G.edges),
                    problem.traffic_matrix.seed,
                    problem.traffic_matrix.model,
                    len(problem.commodity_list),
                    problem.total_demand,
                    "path_formulation",
                    "N/A",
                    "N/A",
                    "N/A",
                    num_paths,
                    edge_disjoint,
                    dist_metric,
                    total_num_fib_entries,
                    max_num_fib_entries,
                )
                print_(result_line, file=results)

            if benchmark_nc:
                (
                    num_paths,
                    edge_disjoint,
                    dist_metric,
                    partition_cls,
                    num_parts_scale_factor,
                ) = NCFLOW_HYPERPARAMS[problem_name]
                problem = get_fib_entry_problem(problem_name)

                num_parts_to_set = num_parts_scale_factor * int(
                    np.sqrt(len(problem.G.nodes))
                )
                partitioner = partition_cls(num_parts_to_set)
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
                total_num_fib_entries, max_num_fib_entries = NcfEpi.fib_entries(
                    problem, partitioner, num_paths, edge_disjoint, dist_metric
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

                result_line = PLACEHOLDER.format(
                    problem_name,
                    len(problem.G.nodes),
                    len(problem.G.edges),
                    problem.traffic_matrix.seed,
                    problem.traffic_matrix.model,
                    len(problem.commodity_list),
                    problem.total_demand,
                    "ncflow_edge_per_iter",
                    partition_algo,
                    num_parts,
                    partitioner.runtime,
                    num_paths,
                    edge_disjoint,
                    dist_metric,
                    total_num_fib_entries,
                    max_num_fib_entries,
                )
                print_(result_line, file=results)

            if benchmark_smore:
                num_paths = 4
                problem = get_fib_entry_problem(problem_name)
                total_num_fib_entries, max_num_fib_entries = SMORE.fib_entries(
                    problem, num_paths
                )
                print_(
                    "SMORE, Problem {}, {} paths, total num fib entries: {}".format(
                        problem_name, num_paths, total_num_fib_entries
                    )
                )
                print_(
                    "SMORE, Problem {}, {} paths, max num fib entries: {}".format(
                        problem_name, num_paths, max_num_fib_entries
                    )
                )

                result_line = PLACEHOLDER.format(
                    problem_name,
                    len(problem.G.nodes),
                    len(problem.G.edges),
                    problem.traffic_matrix.seed,
                    problem.traffic_matrix.model,
                    len(problem.commodity_list),
                    problem.total_demand,
                    "smore",
                    "N/A",
                    "N/A",
                    "N/A",
                    num_paths,
                    "N/A",
                    "N/A",
                    total_num_fib_entries,
                    max_num_fib_entries,
                )
                print_(result_line, file=results)

            if benchmark_fleischer_edge:
                problem = get_fib_entry_problem(problem_name)
                fib_dict = defaultdict(dict)
                for k, (_, t_k, _) in problem.commodity_list:
                    commod_id_str = "k-{}".format(k)
                    # For each commod in a given TM, we would store weights in each switch for each of the
                    # switch's neighbors. For demo purposes, we just store the neighbors. (NOTE: This is
                    # a generous assumption: if the number of neighbors is too large, it cannot fit in a single
                    # rule in the switch.)
                    for n in problem.G.nodes():
                        if n != t_k:
                            fib_dict[n][commod_id_str] = list(problem.G.successors(n))

                fib_dict = dict(fib_dict)
                fib_dict_counts = [len(fib_dict[k]) for k in fib_dict.keys()]
                total_num_fib_entries, max_num_fib_entries = (
                    sum(fib_dict_counts),
                    max(fib_dict_counts),
                )

                print_(
                    "Fleischer Edge, Problem {}, total num fib entries: {}".format(
                        problem_name, total_num_fib_entries
                    )
                )
                print_(
                    "Fleischer Edge, Problem {}, max num fib entries: {}".format(
                        problem_name, max_num_fib_entries
                    )
                )

                result_line = PLACEHOLDER.format(
                    problem_name,
                    len(problem.G.nodes),
                    len(problem.G.edges),
                    problem.traffic_matrix.seed,
                    problem.traffic_matrix.model,
                    len(problem.commodity_list),
                    problem.total_demand,
                    "fleischer_edge",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    total_num_fib_entries,
                    max_num_fib_entries,
                )
                print_(result_line, file=results)


if __name__ == "__main__":
    benchmark(
        PROBLEM_NAMES,
        benchmark_path=True,
        benchmark_nc=True,
        benchmark_smore=True,
        benchmark_fleischer_edge=True,
    )
