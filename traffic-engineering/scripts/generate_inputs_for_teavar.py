#! /usr/bin/env python

import numpy as np
from glob import iglob
import pickle
import os

import sys

sys.path.append("..")

from lib.problem import Problem

OUTPUT_DIR = "../topologies/teavar-format"
TMFOLDERS = ["uniform", "gravity", "bimodal"]

skip_paths = False

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for prob_name in ["Uninett2010.graphml", "b4-teavar.json"]:
        print(prob_name)

        # write out the topoloogy
        PROB_DIR = os.path.join(OUTPUT_DIR, prob_name)
        if not os.path.exists(PROB_DIR):
            os.makedirs(PROB_DIR)

        # read a fake tm as well
        tm_fname = list(
            iglob(
                "../traffic-matrices/{}/{}*_traffic-matrix.pkl".format(
                    TMFOLDERS[0], prob_name
                )
            )
        )[0]
        if prob_name.endswith(".graphml"):
            problem = Problem.from_file(
                "../topologies/topology-zoo/{}".format(prob_name), tm_fname
            )
        else:
            problem = Problem.from_file("../topologies/{}".format(prob_name), tm_fname)

        nodes = dict()
        edges = list()
        edges_to_delete = []
        for u, v, cap in problem.G.edges.data("capacity"):
            if cap == 0.0:
                edges_to_delete.append((u, v))
            else:
                cap_gbps = "{}".format(np.around(cap / 1e3))
                nodes[u] = 1
                nodes[v] = 1
                edges.append((u, v, cap_gbps))
        problem.G.remove_edges_from(edges_to_delete)

        nodes_fname = PROB_DIR + "/nodes.txt"
        with open(nodes_fname, "w") as f:
            print("String_node_names", file=f)
            for u in nodes.keys():
                f.write("s{}\n".format(u + 1))
            f.close()

        topo_fname = PROB_DIR + "/topology.txt"
        with open(topo_fname, "w") as f:
            f.write("to_node\t from_node\t capacity\t prob_failure\n")
            for (u, v, cap) in edges:
                f.write("{}\t {}\t {}\t {}\n".format(u + 1, v + 1, cap, 0.0))
                # f.write("{}\t {}\t {}\t {}\n".format(v, u, cap, 0.0))
            f.close()

        demand_fname = PROB_DIR + "/demand.txt"
        with open(demand_fname, "w") as w:
            for tmtype in TMFOLDERS:
                for tm_fname in iglob(
                    "../traffic-matrices/{}/{}*_traffic-matrix.pkl".format(
                        tmtype, prob_name
                    )
                ):
                    with open(tm_fname, "rb") as f:
                        full_tm = pickle.load(f)
                        for row in full_tm[:-1, :]:
                            print(" ".join(str(x) for x in row), end=" ", file=w)
                        print(" ".join(str(x) for x in full_tm[-1]), end="\n", file=w)
                        f.close()
            w.close()

        if skip_paths:
            continue

        PATHS_DIR = os.path.join(PROB_DIR, "paths")
        if not os.path.exists(PATHS_DIR):
            os.makedirs(PATHS_DIR)

        # paths from SMORE
        for numpaths in [4, 8]:
            path_names = list(
                iglob("../topologies/raecke/{}-{}-paths*".format(prob_name, numpaths))
            )
            if len(path_names) == 0:
                continue
            if len(path_names) > 1:
                print("WARN... multiple matching path files {}".format(path_names))

            path_i_fname = path_names[0]

            paths_dict = {}
            with open(path_i_fname, "r") as f:
                new_src_and_sink = True
                src, target = None, None
                for line in f:
                    line = line.strip()
                    if line == "":
                        new_src_and_sink = True
                        continue

                    if new_src_and_sink:
                        parts = line[:-2].split(" -> ")
                        src, target = int(parts[0][1:]), int(parts[1][1:])
                        paths_dict[(src, target)] = []
                        new_src_and_sink = False
                    else:
                        path = []
                        path_str = line[1 : line.rindex("]")]
                        for edge_str in path_str.split(", "):
                            v = int(edge_str.split(",")[-1][1:-1])
                            path.append(v)
                        paths_dict[(src, target)].append(path)

                f.close()

            path_o_fname = PATHS_DIR + ("/SMORE{}".format(numpaths))
            with open(path_o_fname, "w") as w:
                for (src, target), paths in paths_dict.items():
                    w.write("h{} -> h{} :\n".format(src + 1, target + 1))
                    share = 1.0 / len(paths)
                    for path in paths:
                        w.write("[(h{},s{}), ".format(src + 1, src + 1))
                        prev = src
                        for node in path:
                            if prev != node:
                                w.write("(s{},s{}), ".format(prev + 1, node + 1))
                            prev = node
                        w.write(
                            "(s{},h{})] @ {}\n".format(target + 1, target + 1, share)
                        )
                    w.write("\n")
                w.close()

        # paths from .pkl
        for numpaths in [4, 8]:
            path_names = list(
                iglob(
                    "../topologies/paths/{}-{}-paths_edge-disjoint-True_dist-metric-inv-cap*.pkl".format(
                        prob_name, numpaths
                    )
                )
            )
            if len(path_names) == 0:
                continue
            if len(path_names) > 1:
                print("WARN... multiple matching path files {}".format(path_names))

            path_i_fname = path_names[0]
            paths_dict = {}
            print("Reading paths from {}\n".format(path_i_fname))
            with open(path_i_fname, "rb") as f:
                paths_dict = pickle.load(f)
                for key, paths in paths_dict.items():
                    print("{} = {}".format(key, paths))

                f.close()

            path_o_fname = PATHS_DIR + ("/EDInvCap{}".format(numpaths))
            with open(path_o_fname, "w") as w:
                for (src, target), paths in paths_dict.items():
                    w.write("h{} -> h{} :\n".format(src + 1, target + 1))
                    share = 1.0 / len(paths)
                    for path in paths:
                        w.write("[(h{},s{}), ".format(src + 1, src + 1))
                        prev = src
                        for node in path:
                            if prev != node:
                                w.write("(s{},s{}), ".format(prev + 1, node + 1))
                            prev = node
                        w.write(
                            "(s{},h{})] @ {}\n".format(target + 1, target + 1, share)
                        )
                    w.write("\n")

                w.close()
