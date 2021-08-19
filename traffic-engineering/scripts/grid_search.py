#! /usr/bin/env python

from itertools import product
import numpy as np
import traceback
import os

import sys

sys.path.append("..")

from lib.problem import Problem
from lib.partitioning import FMPartitioning, SpectralClustering
from lib.partitioning.utils import all_partitions_contiguous
from lib.algorithms import NcfEpi
from benchmarks.benchmark_consts import HOLDOUT_PROBLEMS

OUTPUT_CSV = "grid-search.csv"
LOG_DIR = "grid-search-logs"


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()


def grid_search(
    problem_name,
    topo_fname,
    tm_fname,
    num_paths_to_sweep=[4],
    edge_disjoint_to_sweep=[True, False],
    dist_metrics_to_sweep=["inv-cap"],
    partition_algos_to_sweep=["fm_partitioning", "spectral_clustering"],
    num_parts_scale_factors_to_sweep=[1, 2, 3, 4],
):

    problem = Problem.from_file(topo_fname, tm_fname)
    assert problem_name == problem.name
    print_(problem.name, tm_fname)
    traffic_seed = problem.traffic_matrix.seed
    total_demand = np.sum(problem.traffic_matrix.tm)
    print_("traffic seed: {}".format(traffic_seed))
    print_("traffic matrix model: {}".format(problem.traffic_matrix.model))
    print_("traffic scale factor: {}".format(problem.traffic_matrix.scale_factor))
    print_("total demand: {}".format(total_demand))

    num_parts_to_sweep = [
        sf * int(np.sqrt(len(problem.G.nodes)))
        for sf in num_parts_scale_factors_to_sweep
    ]

    for (
        partition_algo,
        num_partitions_to_set,
        num_paths,
        edge_disjoint,
        dist_metric,
    ) in product(
        partition_algos_to_sweep,
        num_parts_to_sweep,
        num_paths_to_sweep,
        edge_disjoint_to_sweep,
        dist_metrics_to_sweep,
    ):
        if partition_algo == "fm_partitioning":
            partitioner = FMPartitioning(num_partitions_to_set)
        elif partition_algo == "spectral_clustering":
            partitioner = SpectralClustering(num_partitions_to_set)

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
            LOG_DIR,
            "ncflow",
            partition_algo,
            "{}-partitions".format(num_partitions_to_set),
            "{}-paths".format(num_paths),
            "edge-disjoint-{}".format(edge_disjoint),
            "{}-dist-metric".format(dist_metric),
        )
        if not os.path.exists(run_nc_dir):
            os.makedirs(run_nc_dir)

        with open(
            os.path.join(
                run_nc_dir,
                "{}-ncflow-{}_partitioner-{}_partitions-{}_paths-{}_edge_disjoint-{}_dist_metric.txt".format(
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
            partition_vector = partitioner.partition(problem)
            if not all_partitions_contiguous(problem, partition_vector):
                print_(
                    "Topology {}, partitioner {}, num_partitions_to_set {} did not find a valid partition".format(
                        topo_fname, partition_algo, num_partitions_to_set
                    )
                )
                continue

            try:
                ncflow = NcfEpi.new_total_flow(
                    num_paths,
                    edge_disjoint=edge_disjoint,
                    dist_metric=dist_metric,
                    out=log,
                )
                ncflow.solve(problem, partitioner)

                num_partitions = len(np.unique(ncflow._partition_vector))
                size_of_largest_partition = partitioner.size_of_largest_partition
                runtime = ncflow.runtime_est(14)
                total_flow = ncflow.obj_val
                with open(OUTPUT_CSV, "a") as w:
                    print_(
                        "{},{},{},{},{},{},{},{},{},{}".format(
                            problem.name,
                            os.path.basename(tm_fname),
                            partition_algo,
                            num_partitions,
                            size_of_largest_partition,
                            num_paths,
                            edge_disjoint,
                            dist_metric,
                            total_flow,
                            runtime,
                        ),
                        file=w,
                    )
            except:
                print_(
                    "TM {}, {} partitioner, {} partitions, {} paths, edge disjoint {}, dist metric {} failed".format(
                        tm_fname,
                        partition_algo,
                        num_partitions_to_set,
                        num_paths,
                        edge_disjoint,
                        dist_metric,
                    )
                )
                traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    with open(OUTPUT_CSV, "a") as w:
        print_(
            "problem,tm_fname,partition_algo,num_partitions,size_of_largest_partition,num_paths,edge_disjoint,dist_metric,total_flow,runtime",
            file=w,
        )

    for problem_name, topo_fname, tm_fname in HOLDOUT_PROBLEMS:
        grid_search(problem_name, topo_fname, tm_fname)
