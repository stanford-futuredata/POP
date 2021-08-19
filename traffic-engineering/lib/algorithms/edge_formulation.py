from .abstract_formulation import AbstractFormulation, Objective
from ..lp_solver import LpSolver
from ..graph_utils import compute_in_or_out_flow
from gurobipy import GRB, Model, quicksum, Var
from collections import defaultdict
import numpy as np
import datetime

EPS = 1e-5
EPS2 = 1e-2

# Given a flow network G(V, E), where each edge (u, v)
# has capacity c(u, v), and a traffic demand matrix (s_k, t_k, d_k) with K non-zero entries
# where s_k is the source, t_k is the target, and d_k is the demand,
# we want to find the flow f_k(u, v), where 0 <= f_k(u, v) <= 1, and represents
# the fraction of the requested flow d_k routed on the edge (u, v).
# We want to maximize the amount of flow through the network:
#  Maximize:
#
#       \alpha // max-min fairness
#       sum_{k}^{K} sum_{v}^{V} (f_k(s_k, v)) // max-flow
#
#  subject to:
#        // demand constraints
#        sum_{v}^{V} (f_k(s_k, v)) <= d_k for all k in K
#        // edge capacity constraints
#        sum_{k}^{K} (f_k(u, v)) <= c(u, v) for all (u, v) \in E
#        // flow conservation on transit nodes
#        sum_{w \in V} f_k(u, w) - sum_{w \in V} f_k(w, u) = 0 for all u \in V - {s_k, t_k} for all k in K
#        // flow from source must equal flow into target
#        sum_{w \in V} f_k(s_k, w) - sum_{w \in V} f_k(w, t_k) = 0 for all k in K
#        // flow into source must be 0
#        sum_{w \in V} f_k(w, s_k) = 0 for all k in K
#        // flow out of target must be 0
#        sum_{w \in V} f_k(t_k, w) = 0 for all k in K
#
#        // constraints for max-min fairness only
#        sum_{v}^{V} (f_k(s_k, v)) >= \alpha for k


class EdgeFormulation(AbstractFormulation):
    @classmethod
    def new_total_flow(cls, out=None):
        return cls(objective=Objective.TOTAL_FLOW, DEBUG=True, VERBOSE=True, out=out)

    def __init__(self, objective, *, DEBUG, VERBOSE, out=None):
        super().__init__(objective, DEBUG, VERBOSE, out)

    def _construct_lp(self, fixed_total_flows=[]):

        m = Model("max-flow: edge-formulation")
        debug_log = False

        # Create variables
        M = len(self.problem.G.edges)  # number of edges
        K = len(self.problem.commodity_list)  # number of commodity flows

        eps = EPS / (
            M
            * max(
                [
                    c_e if c_e else 0.0
                    for _, _, c_e in self.problem.G.edges.data("capacity")
                ]
            )
        )
        # if at least one edge has a non-zero weight, then we need to include the
        # weights as part of our objective function later on
        max_weight = max(
            [w if w else 0.0 for _, _, w in self.problem.G.edges.data("weight")]
        )
        if max_weight > 0.0:
            eps2 = EPS2 / (EPS2 ** -1 + (M * max_weight))

        if debug_log:
            print(
                "to create vars; M(#edges)=",
                M,
                " K(#commods)=",
                K,
                " now: ",
                datetime.datetime.now(),
            )
        self.vars = m.addVars(M, K, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

        if debug_log:
            print("fin: add vars", " now: ", datetime.datetime.now())

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
        if self._objective == Objective.TOTAL_FLOW:
            self._print("TOTAL FLOW objective")
            obj = (
                quicksum(
                    [
                        # TODO: create a mapping instead of using enumerate
                        self.vars[e, k]
                        for k, (_, (s_k, _, d_k)) in enumerate(
                            self.problem.commodity_list
                        )
                        for e, (src, _) in enumerate(self.problem.G.edges())
                        if src == s_k
                    ]
                )
                - eps * self.vars.sum()
            )
            if max_weight > 0.0:
                obj -= eps2 * quicksum(
                    [
                        self.edges_list[e][-1]["weight"] * self.vars[e, k]
                        for e in range(M)
                        for k in range(K)
                    ]
                )
        elif self._objective == Objective.MAX_MIN_FAIRNESS:
            self._print("MAX MIN FAIRNESS objective")
            # constraints for max-min fairness
            # sum_{v}^{V} (f_k(s_k, v)) >= \alpha for k
            self.alpha = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="a")
            m.update()
            for k, (_, (s_k, _, d_k)) in enumerate(self.problem.commodity_list):
                constr_vars = [
                    self.vars[e, k]
                    for e, (src, _) in enumerate(self.problem.G.edges())
                    if src == s_k
                ]
                m.addConstr(quicksum(constr_vars) / d_k >= self.alpha)
            obj = self.alpha

        m.setObjective(obj, GRB.MAXIMIZE)
        if debug_log:
            print("fin: set obj", " now: ", datetime.datetime.now())

        # Edge capacity constraints
        m.addConstrs(
            self.vars.sum(e, "*") <= c_e
            for e, (_, _, c_e) in enumerate(self.problem.G.edges.data("capacity"))
        )
        if debug_log:
            print("fin: cap constr", " now: ", datetime.datetime.now())

        # Demand constraints at src/target, flow conservation constraints
        for k, (_, (src, target, d_k)) in enumerate(self.problem.commodity_list):
            flow_out = defaultdict(list)
            flow_in = defaultdict(list)
            for e, edge in enumerate(self.problem.G.edges()):
                flow_out[edge[0]].append(self.vars[e, k])
                flow_in[edge[1]].append(self.vars[e, k])

            m.addConstr(quicksum(flow_out[src]) <= d_k)
            m.addConstr(quicksum(flow_out[src]) - quicksum(flow_in[target]) == 0)
            m.addConstr(quicksum(flow_in[src]) + quicksum(flow_out[target]) == 0)

            for n in self.problem.G.nodes():
                if n != src and n != target:
                    m.addConstr(quicksum(flow_out[n]) - quicksum(flow_in[n]) == 0)
        if debug_log:
            print(
                "fin: demand cap and flow conservation",
                " now: ",
                datetime.datetime.now(),
            )

        edge_idx = {edge: e for e, edge in enumerate(self.problem.G.edges)}
        for edge, total_flow in fixed_total_flows:
            e = edge_idx[edge]
            m.addConstr(self.vars.sum(e, "*") == total_flow)
        if debug_log:
            print("fin: fixed flows", " now: ", datetime.datetime.now())

        return LpSolver(m, self.debug_fn, self.DEBUG, self.VERBOSE, self.out)

    def extract_sol_as_dict(self, raw_flows=False):

        l = []
        for var in self.model.getVars():
            if var.varName.startswith("f[") and var.x != 0.0:
                e, sol_k = self._extract_inds_from_var_name(var.varName)
                u, v, _ = self.edges_list[e]
                true_k, (s_k, t_k, d_k) = self.problem.commodity_list[sol_k]

                l.append((u, v, true_k, s_k, t_k, d_k, var.x))

        flows_def = defaultdict(list)
        for u, v, k, s_k, t_k, d_k, flow in l:
            edge = (u, v)
            if isinstance(d_k, Var):
                assert d_k.x == flow
            flows_def[(k, (s_k, t_k, d_k))].append((edge, flow))

        # Net-zero flows are set to empty list
        flows = {}
        for commod_key in self.problem.commodity_list:
            _, (s_k, _, _) = commod_key
            flow_list = flows_def[commod_key]
            if raw_flows:
                flows[commod_key] = flow_list
            else:
                src_out_flow = compute_in_or_out_flow(flow_list, 0, {s_k})
                if src_out_flow > 0.0:
                    flows[commod_key] = flow_list
                else:
                    flows[commod_key] = []

        return flows

    def extract_sol_as_mat(self):
        edge_idx = {edge: e for e, edge in enumerate(self.problem.G.edges)}

        l = [
            (self._extract_inds_from_var_name(var.varName), var.x)
            for var in self._solver.model.getVars()
            if var.varName.startswith("f[")
        ]
        inds, vals = zip(*l)
        row_inds, col_inds = zip(*inds)
        mat = np.zeros((max(row_inds) + 1, max(col_inds) + 1))
        for r, c, v in zip(row_inds, col_inds, vals):
            mat[r, c] = v

        for k, (_, (s_k, _, _)) in enumerate(self.problem.commodity_list):
            out_flow = 0.0
            for u in self.problem.G.successors(s_k):
                e = edge_idx[(s_k, u)]
                out_flow += mat[e, k]
            if out_flow == 0.0:
                mat[:, k] = 0.0

        return mat

    @property
    def runtime(self):
        return self._solver.model.Runtime

    @property
    def obj_val(self):
        return self._solver.model.objVal
