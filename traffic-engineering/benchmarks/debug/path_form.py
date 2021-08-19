#! /usr/bin/env python

import pickle
import argparse

import sys

sys.path.append("..")
sys.path.append("../..")

from benchmark_consts import PATH_FORM_HYPERPARAMS
from lib.algorithms import PathFormulation, Objective
from lib.problem import Problem
from lib.graph_utils import check_feasibility


def run_path_form(args):
    topo_fname = args.topo_fname
    tm_fname = args.tm_fname
    obj = args.obj

    problem = Problem.from_file(topo_fname, tm_fname)
    print(problem.name, tm_fname)

    num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS

    pf = PathFormulation(
        objective=Objective.get_obj_from_str(obj),
        num_paths=num_paths,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        DEBUG=True,
    )
    pf.solve(problem)
    print("{}: {}".format(obj, pf.obj_val))
    sol_dict = pf.sol_dict
    check_feasibility(problem, [sol_dict])
    with open("pf-sol-dict.pkl", "wb") as w:
        pickle.dump(sol_dict, w)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, choices=["total_flow", "mcf"], required=True)
    parser.add_argument("--topo_fname", type=str, required=True)
    parser.add_argument("--tm_fname", type=str, required=True)
    args = parser.parse_args()
    run_path_form(args)
