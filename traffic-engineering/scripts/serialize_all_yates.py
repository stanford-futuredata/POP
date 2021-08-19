#! /usr/bin/env python

from benchmarks.benchmark_consts import PROBLEM_NAMES
import networkx as nx
import numpy as np
from glob import iglob
import pickle
import os

import sys

sys.path.append("..")

from lib.problem import Problem

OUTPUT_DIR = "../traffic-matrices/yates-format"
TOPOLOGIES_OUTPUT_DIR = "../topologies/yates-format"
HOSTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "hosts")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(HOSTS_OUTPUT_DIR):
        os.makedirs(HOSTS_OUTPUT_DIR)

    if not os.path.exists(TOPOLOGIES_OUTPUT_DIR):
        os.makedirs(TOPOLOGIES_OUTPUT_DIR)

    for prob_name in PROBLEM_NAMES:
        print(prob_name)

        full_tm = None
        for tm_fname in iglob(
            "../traffic-matrices/full-tms/{}*_traffic-matrix.pkl".format(prob_name)
        ):
            with open(tm_fname, "rb") as f:
                full_tm = pickle.load(f)

        if prob_name.endswith(".graphml"):
            problem = Problem.from_file(
                "../topologies/topology-zoo/{}".format(prob_name), tm_fname
            )
        else:
            problem = Problem.from_file("../topologies/{}".format(prob_name), tm_fname)

        for node, data in problem.G.nodes.data(True):
            keys = list(data.keys())
            for key in keys:
                if key not in ["type", "ip", "mac"]:
                    del problem.G.nodes[node][key]
            problem.G.nodes[node]["type"] = "host"
            problem.G.nodes[node]["ip"] = "111.0.{}.{}".format(node, node)
            problem.G.nodes[node]["mac"] = "00:00:00:00:{}:{}".format(node, node)

        edges_to_delete = []
        for u, v, cap in problem.G.edges.data("capacity"):
            if cap == 0.0:
                edges_to_delete.append((u, v))
            else:
                problem.G[u][v]["capacity"] = "{}Gbps".format(np.around(cap / 1e3))
        problem.G.remove_edges_from(edges_to_delete)

        nx.drawing.nx_agraph.write_dot(
            problem.G,
            os.path.join(
                TOPOLOGIES_OUTPUT_DIR,
                prob_name.replace(".graphml", ".dot").replace(".json", ".dot"),
            ),
        )

        with open(
            os.path.join(
                HOSTS_OUTPUT_DIR,
                prob_name.replace(".graphml", ".hosts").replace(".json", ".hosts"),
            ),
            "w",
        ) as w:
            for node in problem.G.nodes:
                print(node, file=w)

        new_basename = "{}_traffic-matrix.txt".format(prob_name)
        with open(os.path.join(OUTPUT_DIR, new_basename), "w") as w:
            for row in full_tm[:-1, :]:
                print(" ".join(str(x) for x in row), end=" ", file=w)
            print(" ".join(str(x) for x in full_tm[-1]), end="", file=w)
