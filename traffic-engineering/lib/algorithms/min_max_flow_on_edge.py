from .abstract_formulation import Objective
from .edge_formulation import EdgeFormulation
from ..lp_solver import LpSolver
from gurobipy import GRB, Model, quicksum
from collections import defaultdict


class MinMaxFlowOnEdgeOverCap(EdgeFormulation):
    def __init__(self, *, out, DEBUG=False, VERBOSE=False, GAMMA=1e-3):
        super().__init__(
            objective=Objective.MIN_MAX_UTIL, DEBUG=DEBUG, VERBOSE=VERBOSE, out=out
        )
        self.GAMMA = GAMMA

    def _construct_lp(self, fixed_total_flows=[]):

        m = Model("min max flow on edge over cap")

        # Create variables
        M = len(self.problem.G.edges)  # number of edges
        K = len(self.problem.commodity_list)  # number of commodity flows

        vars = m.addVars(M, K, vtype=GRB.CONTINUOUS, lb=0.0, name="f")
        max_per_edge_flow = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="z")
        m.update()
        self.edges_list = list(self.problem.G.edges.data("capacity"))

        if self.DEBUG:
            from functools import partial

            def _debug_fn(e_l, c_l, var):
                e, k = self._extract_inds_from_var_name(var.varName)
                u, v, _ = e_l[e]
                k, (s_k, t_k, d_k) = c_l[k]
                return u, v, k, s_k, t_k, d_k

            self.debug_fn = partial(
                _debug_fn, self.edges_list, self.problem.commodity_list
            )
        else:
            self.debug_fn = None

        # Set objective
        # minimize the maximum flow on edge / capacity
        obj = max_per_edge_flow
        m.setObjective(obj, GRB.MINIMIZE)

        # Max Constraints
        for e, (_, _, c_e) in enumerate(self.problem.G.edges.data("capacity")):
            m.addConstr(vars.sum(e, "*") / c_e <= max_per_edge_flow)

        # Demand constraints at src/target, flow conservation constraints
        for k, (_, (src, target, d_k)) in enumerate(self.problem.commodity_list):
            flow_out = defaultdict(list)
            flow_in = defaultdict(list)
            for e, edge in enumerate(self.problem.G.edges()):
                flow_out[edge[0]].append(vars[e, k])
                flow_in[edge[1]].append(vars[e, k])

            m.addConstr(quicksum(flow_out[src]) == d_k)
            m.addConstr(quicksum(flow_out[src]) - quicksum(flow_in[target]) == 0)
            # Src should have nothing flowing in, target should have nothing flowing out
            m.addConstr(quicksum(flow_in[src]) + quicksum(flow_out[target]) == 0)

            for n in self.problem.G.nodes():
                if n != src and n != target:
                    m.addConstr(quicksum(flow_out[n]) - quicksum(flow_in[n]) == 0)

        edge_idx = {edge: e for e, edge in enumerate(self.problem.G.edges)}
        for edge, total_flow in fixed_total_flows:
            m.addConstr(vars.sum(edge_idx[edge], "*") == total_flow)

        return LpSolver(m, self.debug_fn, self.DEBUG, self.VERBOSE, self.out)
