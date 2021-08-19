#! /usr/bin/env python

import os
import pickle
from pathos import multiprocessing

import sys

sys.path.append("..")

from lib.problems import get_problem
from lib.algorithms.path_formulation import PathFormulation, PATHS_DIR
from lib.path_utils import graph_copy_with_edge_weights, find_paths

global G
global num_paths
global edge_disjoint
global LOAD_FROM_DISK
LOAD_FROM_DISK = True


def find_paths_wrapper(commod):
    k, (s_k, t_k, d_k) = commod
    if LOAD_FROM_DISK:
        if (s_k, t_k) not in paths_dict:
            paths = find_paths(G, s_k, t_k, num_paths, edge_disjoint)
            return ((s_k, t_k), paths)
    else:
        paths = find_paths(G, s_k, t_k, num_paths, edge_disjoint)
        return ((s_k, t_k), paths)


if __name__ == "__main__":
    problem = get_problem(sys.argv[1], model="gravity", random=False)
    assert problem.traffic_matrix.is_full

    global num_paths
    num_paths = int(sys.argv[2])

    dist_metric = sys.argv[3]

    global edge_disjoint
    if sys.argv[4] == "True":
        edge_disjoint = True
    elif sys.argv[4] == "False":
        edge_disjoint = False
    else:
        raise Exception("invalid argument for edge_disjoint: {}".format(sys.argv[4]))

    if not os.path.exists(PATHS_DIR):
        os.makedirs(PATHS_DIR)

    paths_fname = PathFormulation.paths_full_fname(
        problem, num_paths, edge_disjoint, dist_metric
    )

    if LOAD_FROM_DISK:
        print("Loading paths from pickle file", paths_fname)
        try:
            with open(paths_fname, "rb") as f:
                paths_dict = pickle.load(f)
            print("paths_dict: ", len(paths_dict))
        except FileNotFoundError:
            print("Unable to find {}".format(paths_fname))
            paths_dict = {}

    global G
    G = graph_copy_with_edge_weights(problem.G, dist_metric)

    pool = multiprocessing.ProcessPool(28)
    new_paths_dict = pool.map(find_paths_wrapper, problem.commodity_list)
    for ret_val in new_paths_dict:
        if ret_val is not None:
            k, v = ret_val
            paths_dict[k] = v

    print("paths_dict: ", len(paths_dict))
    print("Saving paths to pickle file")
    with open(paths_fname, "wb") as w:
        pickle.dump(paths_dict, w)
