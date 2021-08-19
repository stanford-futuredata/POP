#!/usr/bin/env python

from benchmark_consts import PATH_FORM_HYPERPARAMS, NCFLOW_HYPERPARAMS, print_
from glob import glob
import numpy as np

import sys

sys.path.append("..")

from lib.algorithms import PathFormulation, NcfEpi
from lib.problem import Problem
from lib.graph_utils import compute_in_or_out_flow

RESULTS_FILE_PLACEHOLDER = "demand-tracking-{}-{}-{}.csv"
RESULTS_HEADER = "tm_number,total_demand,satisfied_demand,new_solution,runtime"
TOPOLOGY = "Kdl.graphml"
TM_MODEL = "uniform"
SCALE_FACTOR = 32.0


def generate_sequence_of_tms(
    seed_prob, num_tms, rel_delta_abs_mean, rel_delta_std, seed=1
):
    problems = [seed_prob]
    mean_load = np.mean(seed_prob.traffic_matrix.tm)
    # The mean and std arguments are relative to this traffic matrix;
    # we scale them by the actual mean of the seed TM
    delta_mean = rel_delta_abs_mean * mean_load
    delta_std = rel_delta_std * mean_load

    np.random.seed(seed)
    while len(problems) < num_tms:
        new_prob = problems[-1].copy()
        # Change each demand by delta mean on average, with a spread of delta_std
        perturb_mean = np.random.normal(delta_mean, delta_std)
        perturb_mean *= np.random.choice([-1, 1])
        new_prob.traffic_matrix.perturb_matrix(perturb_mean, delta_std)
        problems.append(new_prob)

    return problems


class PFWarmStart(object):
    def __init__(self):
        num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
        self.pf_warm = PathFormulation.new_total_flow(
            num_paths=num_paths, edge_disjoint=edge_disjoint, dist_metric=dist_metric
        )
        self.pf_warm._warm_start_mode = True
        self.warm_yet = False

    def solve(self, problem):
        if self.warm_yet:
            self.pf_warm.solve_warm_start(problem)
        else:
            self.pf_warm.solve(problem)
            self.warm_yet = True

    @property
    def sol_dict(self):
        return self.pf_warm.sol_dict

    @property
    def name(self):
        return "path_formulation_warm_start"

    @property
    def runtime(self):
        return self.pf_warm.runtime


class PF(object):
    def __init__(self):
        num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
        self.pf = PathFormulation.new_total_flow(
            num_paths=num_paths, edge_disjoint=edge_disjoint, dist_metric=dist_metric
        )

    def solve(self, problem):
        self.pf.solve(problem)

    @property
    def sol_dict(self):
        return self.pf.sol_dict

    @property
    def name(self):
        return "path_formulation"

    @property
    def runtime(self):
        return self.pf.runtime


class NcI(object):
    def __init__(self, G):
        num_paths, edge_disjoint, dist_metric, partition_cls, sf = NCFLOW_HYPERPARAMS[
            problems[0].name
        ]
        num_parts = sf * int(np.sqrt(len(G)))
        self.ncflow = NcfEpi.new_total_flow(
            num_paths=num_paths, edge_disjoint=edge_disjoint, dist_metric=dist_metric
        )
        self.partitioner = partition_cls(num_parts)

    def solve(self, problem):
        self.ncflow.solve(problem, self.partitioner)

    @property
    def sol_dict(self):
        return self.ncflow.sol_dict

    @property
    def name(self):
        return "ncflow_edge_per_iter"

    @property
    def runtime(self):
        return self.ncflow.runtime_est(14)


def get_algo(arg, G):
    if arg == "--path-form":
        return PF()
    elif arg == "--pfws":
        return PFWarmStart()
    elif arg == "--ncflow":
        return NcI(G)
    else:
        raise Exception('invalid argument "{}"'.format(arg))


def compute_satisfied_demand(problem, sol_dict, residual_factor):
    if sol_dict is None:
        return 0.0, problem.traffic_matrix.tm.copy() * residual_factor

    curr_demand_dict = {
        (s_k, t_k): d_k for _, (s_k, t_k, d_k) in problem.commodity_list
    }
    total_demand_satisfied = 0.0
    residual_tm = np.zeros_like(problem.traffic_matrix.tm)

    for (_, (s_k, t_k, _)), flow_list in sol_dict.items():
        if (s_k, t_k) not in curr_demand_dict:
            continue
        curr_demand = curr_demand_dict[(s_k, t_k)]
        total_flow = compute_in_or_out_flow(flow_list, 0, {s_k})
        # If the current problem requests less flow than what we solved for,
        # we only provide the flow we solved for
        real_flow = min(curr_demand, total_flow)
        total_demand_satisfied += real_flow
        residual_demand = curr_demand - real_flow
        residual_tm[s_k, t_k] = residual_demand

    return total_demand_satisfied, residual_tm * residual_factor


def demand_tracking(
    algo, problems, time_per_prob, residual_factor, results_file_placeholder
):
    results_fname = results_file_placeholder.format(
        problems[0].name, residual_factor, algo.name
    )
    print(results_fname)
    with open(results_fname, "w") as w:
        print_(RESULTS_HEADER, file=w)

        i = 0
        curr_sol_dict = None

        while i < len(problems):
            problem = problems[i]
            print_("\nProblem {}".format(i))
            algo.solve(problem)
            runtime = algo.runtime
            while runtime > time_per_prob and i < len(problems):
                satisfied_demand, residual_tm = compute_satisfied_demand(
                    problems[i], curr_sol_dict, residual_factor
                )
                print_(
                    "{},{},{},False,NaN".format(
                        i, problems[i].total_demand, satisfied_demand
                    ),
                    file=w,
                )
                runtime -= time_per_prob
                i += 1
                if i < len(problems):
                    problems[i].traffic_matrix.tm += residual_tm

            if i >= len(problems):
                break
            curr_sol_dict = algo.sol_dict
            satisfied_demand, residual_tm = compute_satisfied_demand(
                problems[i], curr_sol_dict, residual_factor
            )
            print_(
                "{},{},{},True,{}".format(
                    i, problems[i].total_demand, satisfied_demand, algo.runtime
                ),
                file=w,
            )
            i += 1
            if i >= len(problems):
                break
            problems[i].traffic_matrix.tm += residual_tm


def print_delta_norms(problems):
    first_prob = problems[0]
    prev_tm = first_prob.traffic_matrix.tm
    prev_norm = np.linalg.norm(prev_tm)

    delta_norm_norms = []
    for problem in problems[1:]:
        tm = problem.traffic_matrix.tm
        delta_tm = tm - prev_tm
        delta_norm = np.linalg.norm(delta_tm)
        delta_norm_norm = delta_norm / prev_norm
        print(delta_norm_norm)
        delta_norm_norms.append(delta_norm_norm)
        prev_tm = tm
        prev_norm = np.linalg.norm(tm)

    print("Mean:", np.mean(delta_norm_norms))


def print_total_demand(problems):
    total_demands = []
    for problem in problems:
        print(problem.total_demand)
        total_demands.append(problem.total_demand)

    print("Mean:", np.mean(total_demands))


def print_entry_demands(problems, nodelist):
    assert len(nodelist) == 2
    i = nodelist[0]
    j = nodelist[1]
    print("Demands for {}->{}".format(i, j))
    for problem in problems:
        print(problem.traffic_matrix.tm[i, j])


if __name__ == "__main__":
    seed_tm_fname = glob(
        "../traffic-matrices/{}/{}_{}_*_{}_*.pkl".format(
            TM_MODEL, TOPOLOGY, TM_MODEL, SCALE_FACTOR
        )
    )[0]
    seed_prob = Problem.from_file(
        "../topologies/topology-zoo/{}".format(TOPOLOGY), seed_tm_fname
    )
    num_tms = 25
    time_per_prob = 5 * 60

    problems = generate_sequence_of_tms(
        seed_prob, num_tms, rel_delta_abs_mean=0.25, rel_delta_std=0.5
    )

    if sys.argv[-1] == "--delta-norms":
        print_delta_norms(problems)
    elif sys.argv[-1] == "--total-demands":
        print_total_demand(problems)
    elif sys.argv[-1] == "--print-an-entry":
        if len(sys.argv) == 4:
            entry_tuple = np.array([int(sys.argv[1]), int(sys.argv[2])])
        else:
            entry_tuple = np.random.randint(len(problems[0].G.nodes), size=2)
        print_entry_demands(problems, entry_tuple)
    else:
        residual_factor = float(sys.argv[1])
        results_file_placeholder = RESULTS_FILE_PLACEHOLDER
        algo = get_algo(sys.argv[2], problems[0].G)
        if algo.name == "path_formulation" and sys.argv[-1] == "--oracle":
            time_per_prob = sys.maxsize
            results_file_placeholder = results_file_placeholder.replace(
                ".csv", "-oracle.csv"
            )

        print("Algo: {}".format(algo.name))
        demand_tracking(
            algo, problems, time_per_prob, residual_factor, results_file_placeholder
        )
