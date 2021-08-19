#! /usr/bin/env python

from benchmark_consts import print_, PROBLEM_NAMES, TM_MODELS, NCFLOW_HYPERPARAMS
import numpy as np
import os
from glob import iglob

import sys

sys.path.append("..")
from lib.problem import Problem

OUTPUT_CSV = "demand-stats.csv"
HEADERS = [
    "problem",
    "num_nodes",
    "num_edges",
    "traffic_seed",
    "tm_model",
    "scale_factor",
    "num_commodities",
    "total_demand",
    "clustering_algo",
    "num_partitions",
    "size_of_largest_partition",
    "partition_runtime",
    "intra_demand",
    "inter_demand",
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)
PARTITIONER_DICT = {}

if __name__ == "__main__":
    with open(OUTPUT_CSV, "a") as w:
        print_(",".join(HEADERS), file=w)
        for problem_name in PROBLEM_NAMES:

            if problem_name.endswith(".graphml"):
                topo_fname = os.path.join(
                    "..", "topologies", "topology-zoo", problem_name
                )
            else:
                topo_fname = os.path.join("..", "topologies", problem_name)
            _, _, _, partition_cls, num_parts_scale_factor = NCFLOW_HYPERPARAMS[
                problem_name
            ]

            for model in TM_MODELS:
                for tm_fname in iglob(
                    "../traffic-matrices/{}/{}*_traffic-matrix.pkl".format(
                        model, problem_name
                    )
                ):
                    print(tm_fname)
                    vals = os.path.basename(tm_fname)[:-4].split("_")
                    _, traffic_seed, scale_factor = (
                        vals[1],
                        int(vals[2]),
                        float(vals[3]),
                    )
                    problem = Problem.from_file(topo_fname, tm_fname)

                    if problem_name not in PARTITIONER_DICT:
                        num_partitions_to_set = num_parts_scale_factor * int(
                            np.sqrt(len(problem.G.nodes))
                        )
                        partitioner = partition_cls(num_partitions_to_set)
                        PARTITIONER_DICT[problem_name] = partitioner
                    else:
                        partitioner = PARTITIONER_DICT[problem_name]
                    partition_algo = partitioner.name

                    intra_demand, inter_demand = problem.intra_and_inter_demands(
                        partitioner
                    )
                    result_line = PLACEHOLDER.format(
                        problem.name,
                        len(problem.G.nodes),
                        len(problem.G.edges),
                        traffic_seed,
                        problem.traffic_matrix.model,
                        problem.traffic_matrix.scale_factor,
                        len(problem.commodity_list),
                        problem.total_demand,
                        partition_algo,
                        partitioner.num_partitions,
                        partitioner.size_of_largest_partition,
                        partitioner.runtime,
                        intra_demand,
                        inter_demand,
                    )
                    print_(result_line, file=w)
