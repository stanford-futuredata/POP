#! /usr/bin/env python

from pathos import multiprocessing
import argparse
import os
import pickle

import sys

sys.path.append("..")

from lib.config import TL_DIR
from benchmarks.benchmark_consts import get_problems
from lib.problem import Problem
from lib.algorithms.path_formulation import PathFormulation

OUTPUT_DIR_ROOT = os.path.join(TL_DIR, "traffic-matrices")


def serialize_problem(prob, fname):
    with open(fname, "w") as w:
        print(len(prob.G.nodes), file=w)
        for node in prob.G.nodes:
            print(node, 0, 0, file=w)

        print(len(prob.G.edges), file=w)
        for e, (u, v, c_e) in enumerate(prob.G.edges.data("capacity")):
            print(e, u, v, c_e, 1.0, file=w)

        print(len(prob.commodity_list), file=w)
        for k, (s_k, t_k, d_k) in prob.commodity_list:
            print(k, s_k, t_k, d_k, file=w)


def serialize_paths(fname, paths_dict_fname):
    with open(paths_dict_fname, "rb") as f:
        paths_dict = pickle.load(f)

    with open(fname, "a") as w:
        for (src, target), paths in paths_dict.items():
            print("{} -> {}".format(src, target), file=w)
            for path in paths:
                print("[" + ",".join(str(x) for x in path) + "]", file=w)
            print(file=w)


def serialize(run_args):
    prob_name, topo_fname, tm_fname, output_fname, cmd_args = run_args
    print(prob_name, topo_fname, tm_fname)
    if cmd_args.paths:
        num_paths = args.num_paths
        edge_disjoint = args.edge_disjoint
        dist_metric = args.dist_metric

    prob = Problem.from_file(topo_fname, tm_fname)
    serialize_problem(prob, output_fname)

    if cmd_args.paths:
        paths_dict_fname = PathFormulation.paths_full_fname(
            prob, num_paths, edge_disjoint, dist_metric
        )
        serialize_paths(output_fname, paths_dict_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # create the parser for the "paths" arg
    parser_path = subparsers.add_parser("path")
    parser_path.set_defaults(paths=True)
    parser_path.add_argument("--num-paths", type=int, required=True)
    parser_path.add_argument("--edge-disjoint", type=bool, required=True)
    parser_path.add_argument(
        "--dist-metric", type=str, choices=("inv-cap", "min-hop"), required=True
    )

    # create the parser for the "edge" command
    parser_edge = subparsers.add_parser("edge")
    parser_edge.set_defaults(paths=False)

    args = parser.parse_args()
    if args.paths:
        output_dir_tl = os.path.join(OUTPUT_DIR_ROOT, "fleischer-with-paths-format")
        fname_placeholder = (
            "{}_"
            + "{}-paths_edge_disjoint-{}_dist_metric-{}.txt".format(
                args.num_paths, args.edge_disjoint, args.dist_metric
            )
        )
    else:
        output_dir_tl = os.path.join(OUTPUT_DIR_ROOT, "fleischer-edge-format")
        fname_placeholder = "{}.txt"

    run_args = []
    for slice in range(5):
        args.slices = [slice]
        problems = get_problems(args)
        output_dir = os.path.join(output_dir_tl, "slice-{}".format(slice))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for prob_name, topo_fname, tm_fname in problems:
            output_fname = os.path.join(
                output_dir, fname_placeholder.format(os.path.basename(tm_fname))
            )
            run_args.append((prob_name, topo_fname, tm_fname, output_fname, args))

    pool = multiprocessing.ProcessPool(14)
    pool.map(serialize, run_args)
