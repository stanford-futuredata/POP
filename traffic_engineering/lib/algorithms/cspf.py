from ..lp_solver import LpSolver
from ..graph_utils import path_to_edge_list
from ..path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
from ..config import TOPOLOGIES_DIR
from .abstract_formulation import AbstractFormulation, Objective
from .path_formulation import PathFormulation
from gurobipy import GRB, Model, quicksum
from collections import defaultdict
from datetime import datetime
import numpy as np
import re
import os
import time
import pickle

PATHS_DIR = os.path.join(TOPOLOGIES_DIR, "paths", "path-form")


class CSPF(AbstractFormulation):
    @classmethod
    def new_total_flow(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None
    ):
        return cls(
            objective=Objective.TOTAL_FLOW,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def new_max_concurrent_flow(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None
    ):
        return cls(
            objective=Objective.MAX_CONCURRENT_FLOW,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def new_min_max_link_util(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None
    ):
        return cls(
            objective=Objective.MIN_MAX_LINK_UTIL,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def compute_demand_scale_factor(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None
    ):
        return cls(
            objective=Objective.COMPUTE_DEMAND_SCALE_FACTOR,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    def __init__(
        self,
        *,
        objective,
        num_paths,
        edge_disjoint,
        dist_metric,
        DEBUG=False,
        VERBOSE=False,
        out=None
    ):
        super().__init__(objective, DEBUG, VERBOSE, out)
        if dist_metric != "inv-cap" and dist_metric != "min-hop":
            raise Exception(
                'invalid distance metric: {}; only "inv-cap" and "min-hop" are valid choices'.format(
                    dist_metric
                )
            )
        self._num_paths = num_paths
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric

    ###############################
    # Override superclass methods #
    ###############################

    # TODO: Keep track explicitly of flow paths,
    # so that we can construct the sol_dict
    def solve(self, problem):
        com_list = problem.commodity_list
        tm = problem.traffic_matrix.tm

        paths_dict = PathFormulation.read_paths_from_disk_or_compute(
            problem, self._num_paths, self.edge_disjoint, self.dist_metric,
        )

        # initialize link capacity dict
        remaining_link_capacity_dict = {}
        for u, v in problem.G.edges:
            remaining_link_capacity_dict[(u, v)] = problem.G[u][v]["capacity"]

        # sort paths in ascending order
        all_paths_list = []
        allocated_coms = {}
        for _, (source, target, demand) in com_list:
            paths_array = paths_dict[(source, target)]
            all_paths_list += paths_array
            allocated_coms[(source, target)] = False
        all_paths_list.sort(key=len)

        # iterate through sorted paths
        self._total_flow = 0
        startTime = datetime.now()
        for path in all_paths_list:
            source = path[0]
            target = path[-1]
            demand = tm[source, target]

            # skip if we have already allocated this commodity
            if allocated_coms[(source, target)]:
                continue

            # check that each edge in list has enough capacity
            edge_list = list(path_to_edge_list(path))
            room = True
            for u, v in edge_list:
                if remaining_link_capacity_dict[(u, v)] < demand:
                    room = False
                    break

            if not room:
                continue

            # allocate
            for u, v in edge_list:
                remaining_link_capacity_dict[(u, v)] -= demand
            allocated_coms[(source, target)] = True
            self._total_flow += demand

        self._runtime = (datetime.now() - startTime).total_seconds()

    @property
    def obj_val(self):
        return self._total_flow

    @property
    def runtime(self):
        return self._runtime
