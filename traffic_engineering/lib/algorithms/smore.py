from ..lp_solver import LpSolver
from ..graph_utils import path_to_edge_list, compute_in_or_out_flow
from ..path_utils import remove_cycles
from ..config import TOPOLOGIES_DIR
from .abstract_formulation import AbstractFormulation, Objective
from gurobipy import GRB, Model, quicksum
from collections import defaultdict
import numpy as np
import re
import os
import pickle
import sys

PATHS_DIR = os.path.join(TOPOLOGIES_DIR, "paths", "raeke")


# TODO: this should probably be a subclass of PathFormulation, but not
# necessary to change now
class SMORE(AbstractFormulation):
    @classmethod
    def new_max_link_util(cls, num_paths, out=sys.stdout):
        return cls(
            objective=Objective.MIN_MAX_LINK_UTIL,
            num_paths=num_paths,
            DEBUG=True,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def new_total_flow(cls, num_paths, out=sys.stdout):
        return cls(
            objective=Objective.TOTAL_FLOW,
            num_paths=num_paths,
            DEBUG=True,
            VERBOSE=False,
            out=out,
        )

    def __init__(self, *, objective, num_paths, DEBUG, VERBOSE, out=None):
        super().__init__(objective, DEBUG, VERBOSE, out)
        self._num_paths = num_paths

    def _construct_total_flow_lp(self, G, edge_to_paths, num_total_paths):
        m = Model("total-flow")

        # Create variables: one for each path
        path_vars = m.addVars(num_total_paths, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

        obj = quicksum(path_vars)
        m.setObjective(obj, GRB.MAXIMIZE)

        # Add demand constraints
        commod_id_to_path_inds = {}
        for k, d_k, path_inds in self.commodities:
            commod_id_to_path_inds[k] = path_inds
            m.addConstr(quicksum(path_vars[p] for p in path_inds) <= d_k)

        # Add edge capacity constraints
        for u, v, c_e in G.edges.data("capacity"):
            paths = edge_to_paths[(u, v)]
            constr_vars = [path_vars[p] for p in paths]
            m.addConstr(quicksum(constr_vars) <= c_e)

        return LpSolver(m, None, self.DEBUG, self.VERBOSE, self.out)

    def _construct_smore_lp(self, G, edge_to_paths, num_total_paths):
        m = Model("min-edge-util")

        # Create variables: one for each path
        path_vars = m.addVars(
            num_total_paths, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="f"
        )
        max_link_util_var = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="z")
        m.update()
        m.setObjective(max_link_util_var, GRB.MINIMIZE)

        # Add demand constraints
        for k, d_k, path_ids in self.commodities:
            m.addConstr(quicksum(path_vars[p] for p in path_ids) == 1)

        # Add edge util constraints
        for u, v, c_e in G.edges.data("capacity"):
            paths = edge_to_paths[(u, v)]
            constr_vars = [
                path_vars[p] * self.commodities[self._path_to_commod[p]][-2]
                for p in paths
            ]
            m.addConstr(quicksum(constr_vars) / c_e <= max_link_util_var)

        return LpSolver(m, None, self.DEBUG, self.VERBOSE, self.out)

    @staticmethod
    def paths_full_fname_txt(problem, num_paths):
        return os.path.join(
            PATHS_DIR, "{}-{}-paths-rrt.txt".format(problem.name, num_paths)
        )

    @staticmethod
    def paths_full_fname_pkl(problem, num_paths):
        return os.path.join(
            PATHS_DIR, "{}-{}-paths-rrt.pkl".format(problem.name, num_paths)
        )

    ###############################
    # Override superclass methods #
    ###############################

    def pre_solve(self, problem=None):
        if problem is None:
            problem = self.problem
        paths_fname_txt = SMORE.paths_full_fname_txt(problem, self._num_paths)
        paths_fname_pkl = SMORE.paths_full_fname_pkl(problem, self._num_paths)

        if os.path.exists(paths_fname_pkl):
            self._print("Loading Raeke paths from pickle file", paths_fname_pkl)
            with open(paths_fname_pkl, "rb") as r:
                paths_dict = pickle.load(r)
        else:
            self._print("Loading Raeke paths from text file", paths_fname_txt)
            try:
                with open(paths_fname_txt, "r") as f:
                    new_src_and_sink = True
                    src, target = None, None
                    paths_dict = {}
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
                            path = [src]
                            path_str = line[1 : line.rindex("]")]
                            for edge_str in path_str.split(", "):
                                v = int(edge_str.split(",")[-1][1:-1])
                                path.append(v)

                            paths_dict[(src, target)].append(remove_cycles(path))

                self._print("Saving Raeke paths to pickle file")
                with open(paths_fname_pkl, "wb") as w:
                    pickle.dump(paths_dict, w)

            except FileNotFoundError as e:
                self._print("Unable to find {}".format(paths_fname_txt))
                raise e

        self.commodities = []
        edge_to_paths = defaultdict(list)
        self._path_to_commod = {}
        self._all_paths = []

        path_i = 0
        for k, (s_k, t_k, d_k) in problem.commodity_list:
            paths = paths_dict[(s_k, t_k)]
            path_ids = []
            for path in paths:
                self._all_paths.append(path)

                for edge in path_to_edge_list(path):
                    edge_to_paths[edge].append(path_i)
                path_ids.append(path_i)

                self._path_to_commod[path_i] = k
                path_i += 1

            self.commodities.append((k, d_k, path_ids))
        self._print("pre_solve done")
        return edge_to_paths, path_i

    def _construct_lp(self, sat_flows=[]):
        edge_to_paths, num_paths = self.pre_solve()
        if self._objective == Objective.TOTAL_FLOW:
            self._print("Constructing Total Flow LP")
            return self._construct_total_flow_lp(
                self.problem.G, edge_to_paths, num_paths
            )
        elif self._objective == Objective.MIN_MAX_LINK_UTIL:
            self._print("Constructing SMORE LP")
            return self._construct_smore_lp(self.problem.G, edge_to_paths, num_paths)

    @property
    def sol_dict(self):
        if not hasattr(self, "_sol_dict"):
            sol_dict_def = defaultdict(list)
            for var in self.model.getVars():
                if var.varName.startswith("f[") and var.x != 0.0:
                    match = re.match(r"f\[(\d+)\]", var.varName)
                    p = int(match.group(1))
                    commod_key = self.problem.commodity_list[self._path_to_commod[p]]
                    d_k = commod_key[-1][-1]
                    flow_val = (
                        var.x * d_k
                        if self._objective == Objective.MIN_MAX_LINK_UTIL
                        else var.x
                    )
                    sol_dict_def[commod_key] += [
                        (edge, flow_val)
                        for edge in path_to_edge_list(self._all_paths[p])
                    ]

            # Set zero-flow commodities to be empty lists
            self._sol_dict = {}
            sol_dict_def = dict(sol_dict_def)
            for commod_key in self.problem.commodity_list:
                if commod_key in sol_dict_def:
                    self._sol_dict[commod_key] = sol_dict_def[commod_key]
                else:
                    self._sol_dict[commod_key] = []

        return self._sol_dict

    @property
    def sol_mat(self):
        edge_idx = self.problem.edge_idx
        sol_mat = np.zeros((len(edge_idx), len(self._path_to_commod)), dtype=np.float32)
        for var in self.model.getVars():
            if var.varName.startswith("f[") and var.x != 0.0:
                match = re.match(r"f\[(\d+)\]", var.varName)
                p = int(match.group(1))
                commod_key = self.problem.commodity_list[self._path_to_commod[p]]
                k, d_k = commod_key[0], commod_key[-1][-1]
                for edge in path_to_edge_list(self._all_paths[p]):
                    sol_mat[edge_idx[edge], k] += var.x * d_k

        return sol_mat

    @property
    def total_flow(self):
        if self._objective == Objective.TOTAL_FLOW:
            return self.obj_val
        else:
            sol_dict = self.sol_dict()
            total_flow = 0.0
            for commod_key in self.problem.commodity_list:
                flow_list = sol_dict[commod_key]

                flow = compute_in_or_out_flow(flow_list, 0, {commod_key[-1][0]})
                assert flow <= commod_key[-1][-1]
                total_flow += flow

            return total_flow

    @classmethod
    # Return total number of fib entries and max for any node in topology
    # NOTE: problem has to have a full TM matrix; otherwise, our path set
    # will have a problem
    def fib_entries(cls, problem, num_paths):
        assert problem.is_traffic_matrix_full
        smore = cls.new_max_link_util(num_paths=num_paths)
        smore.pre_solve(problem)
        return smore.num_fib_entries_for_path_set()

    # Return (total # of fib entries, max # of fib entries)
    def num_fib_entries_for_path_set(self):
        self.fib_dict = defaultdict(dict)
        for k, _, path_ids in self.commodities:
            commod_id_str = "k-{}".format(k)
            src = list(path_to_edge_list(self._all_paths[path_ids[0]]))[0][0]
            # For a given TM, we would store weights for each path id. For demo
            # purposes, we just store the path ids
            self.fib_dict[src][commod_id_str] = path_ids

            for path_id in path_ids:
                for u, v in path_to_edge_list(self._all_paths[path_id]):
                    assert path_id not in self.fib_dict[u]
                    self.fib_dict[u][path_id] = v

        self.fib_dict = dict(self.fib_dict)
        fib_dict_counts = [len(self.fib_dict[k]) for k in self.fib_dict.keys()]
        return sum(fib_dict_counts), max(fib_dict_counts)

    @property
    def runtime(self):
        return self._solver.model.Runtime

    @property
    def obj_val(self):
        return self._solver.model.objVal
