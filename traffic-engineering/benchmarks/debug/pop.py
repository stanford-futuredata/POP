#! /usr/bin/env python

import pickle
import argparse

import sys

sys.path.append("..")
sys.path.append("../..")

from benchmark_consts import PATH_FORM_HYPERPARAMS
from lib.algorithms import POP, Objective
from lib.problem import Problem
from lib.graph_utils import check_feasibility


def run_pop(args):
    obj = args.obj
    topo_fname = args.topo_fname
    tm_fname = args.tm_fname
    num_subproblems = args.num_subproblems
    split_method = args.split_method
    split_fraction = args.split_fraction

    problem = Problem.from_file(topo_fname, tm_fname)
    print(problem.name, tm_fname)

    num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS

    pop = POP(
        objective=Objective.get_obj_from_str(obj),
        num_subproblems=num_subproblems,
        split_method=split_method,
        split_fraction=split_fraction,
        num_paths=num_paths,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        DEBUG=True,
    )
    pop.solve(problem)
    print("{}: {}".format(obj, pop.obj_val))
    sol_dict = pop.sol_dict
    check_feasibility(problem, [sol_dict])
    with open("pop-sol-dict.pkl", "wb") as w:
        pickle.dump(sol_dict, w)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, choices=["total_flow", "mcf"], required=True)
    parser.add_argument("--topo_fname", type=str, required=True)
    parser.add_argument("--tm_fname", type=str, required=True)
    parser.add_argument(
        "--num_subproblems", type=int, choices=[1, 2, 4, 8, 16, 32, 64], required=True
    )
    parser.add_argument(
        "--split_method",
        type=str,
        choices=["random", "tailored", "means"],
        required=True,
    )
    parser.add_argument("--split_fraction", type=float, required=True)
    args = parser.parse_args()
    run_pop(args)
