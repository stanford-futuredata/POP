import os
import numpy as np
import sys

sys.path.append("..")

from lib.algorithms.path_formulation import PathFormulation as PF
from lib.problem import Problem
from lib.traffic_matrix import GenericTrafficMatrix
from lib.config import TOPOLOGIES_DIR
from benchmarks.benchmark_consts import NCFLOW_HYPERPARAMS
from lib.algorithms import NcfEpi
from lib.partitioning import FMPartitioning, SpectralClustering

import datetime

t_arg = sys.argv[1]

# read the topology
if t_arg.endswith(".graphml"):
    topo_fname = os.path.join(TOPOLOGIES_DIR, "topology-zoo/" + t_arg)
else:
    topo_fname = os.path.join(TOPOLOGIES_DIR, t_arg)

if t_arg.endswith(".json"):
    G = Problem._read_graph_json(topo_fname)
elif t_arg.endswith(".graphml"):
    G = Problem._read_graph_graphml(topo_fname)
num_nodes = len(G.nodes)
print("#nodes={}".format(num_nodes))

# process each traffic matrix
TEAVAR_DEMANDS_DIR = "../code/teavar/code/data"
d_fname = os.path.join(TEAVAR_DEMANDS_DIR, t_arg, "demand.txt")

line_num = 0
with open(d_fname, "r") as input_file:
    for line in input_file:
        line_num = line_num + 1
        # if line_num != 7:
        #   continue
        print("==================Demand {}==================".format(line_num))
        tm = GenericTrafficMatrix(
            problem=None,
            tm=np.fromstring(line, np.float32, sep=" ").reshape(num_nodes, num_nodes),
        )

        # if line_num == 7:
        #    print("tm=[{}]".format(tm.tm))

        p = Problem(G, tm)
        p.name = t_arg

        # compute demand scale factor
        pf_cdsf = PF.compute_demand_scale_factor(4, edge_disjoint=True)
        pf_cdsf.solve(p)
        z = pf_cdsf.obj_val

        # compute pf solution and runtime
        pf = PF.new_total_flow(4, edge_disjoint=True)
        pf.solve(p)
        pf_flow = pf.obj_val
        pf_runtime = pf.runtime

        # compute nc solution and runtime
        # print("---> p.name = {}".format(p.name))
        if p.name in NCFLOW_HYPERPARAMS:
            (
                num_paths,
                edge_disjoint,
                dist_metric,
                partition_algo,
                sf,
            ) = NCFLOW_HYPERPARAMS[p.name]
            num_partitions = sf * int(np.sqrt(len(p.G.nodes)))

            # print("---> partition_algo = {}".format(partition_algo))
            if False:
                if partition_algo.contains("spectral_clustering"):
                    partition_cls = SpectralClustering
                elif partition_algo.contains("fm_partitioning"):
                    partition_cls = FMPartitioning
                else:
                    print(
                        "WARN un-parseable partition_algo = {}".format(partition_algo)
                    )

            partitioner = partition_algo(num_partitions)

            ncflow = NcfEpi.new_total_flow(
                num_paths, edge_disjoint=True, dist_metric="inv-cap"
            )
            begin = datetime.datetime.now()
            ncflow.solve(p, partitioner)
            end = datetime.datetime.now()

            nc_flow = ncflow.obj_val
            nc_runtime = ncflow.runtime_est(14)
            nc_wallclocktime = (end - begin).seconds
        else:
            nc_flow = pf_flow
            nc_runtime = pf_runtime
            nc_wallclocktime = -1

        print(
            "RESULT D {0} (paths=edinvcap4) z {1:1.3f} PF flow/runtime {2:1.3f} {3:1.3f} NCFlow flow/runtime/wc {4:1.3f} {5:1.3f} {6:1.3f}\n".format(
                line_num, z, pf_flow, pf_runtime, nc_flow, nc_runtime, nc_wallclocktime
            )
        )

        # quit()
