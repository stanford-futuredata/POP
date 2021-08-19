from ..abstract_formulation import Objective
from .ncflow_abstract import NCFlowAbstract
from ...graph_utils import (
    compute_in_or_out_flow,
    get_in_and_out_neighbors,
    path_to_edge_list,
    neighbors_and_flows,
    assert_flow_conservation,
)
from ...lp_solver import LpSolver, Method
from ...utils import waterfall_memoized

from gurobipy import GRB, Model, quicksum
from collections import defaultdict
from itertools import product

import networkx as nx
import numpy as np

import re
import sys


EPS = 1e-5


class NCFlowSingleIter(NCFlowAbstract):
    @classmethod
    def new_total_flow(cls, out=None):
        if out is None:
            out = sys.stdout
        return cls(objective=Objective.TOTAL_FLOW, DEBUG=False, VERBOSE=False, out=out)

    def __init__(self, *, objective, DEBUG, VERBOSE, out):
        super().__init__(objective, DEBUG=DEBUG, VERBOSE=VERBOSE, out=out)
        self.r2_min_max_util = True

    ###############
    # EXTRACT SOL #
    ###############

    def extract_reconciliation_sol_as_dict(
        self, model, meta_commodity_list, edges_list
    ):
        l = []
        for var in model.getVars():
            if var.varName.startswith("f[") and var.x > EPS:
                e, sol_k = self._extract_inds_from_var_name(var.varName)
                u, v = edges_list[e]
                k, (s_k, t_k, d_k) = meta_commodity_list[sol_k]

                l.append((u, v, k, s_k, t_k, d_k, var.x))

        sol_dict_def = defaultdict(list)
        for u, v, k, s_k, t_k, d_k, flow in l:
            edge = (u, v)
            sol_dict_def[(k, (s_k, t_k, d_k))].append((edge, flow))

        # Net-zero flows are set to empty list
        return self._create_sol_dict(sol_dict_def, meta_commodity_list)

    def extract_r2_sol_as_dict(
        self, model, multi_commodity_list, intra_commodity_list, all_paths
    ):

        meta_sol_dict_def = defaultdict(list)
        intra_sol_dict_def = defaultdict(list)
        r2_srcs_out_flow_lists_def = defaultdict(list)
        r2_targets_in_flow_lists_def = defaultdict(list)
        mc_id_to_path_id_to_flow = defaultdict(dict)

        # generate the list of meta commodities that are active at this meta-node
        if self.VERBOSE:
            self._print("--> r2mcl: ", multi_commodity_list)
            self._print("--> ci2mci: ", self.commod_id_to_meta_commod_id)

        active_meta_commodity_dict = {}  # active k_meta --> meta_commod_key
        active_multi_commodity_srcs_dict = {}
        active_multi_commodity_targets_dict = {}

        for mc_id, (srcs, targets, _, commod_list) in enumerate(multi_commodity_list):
            srcs_are_virtual = srcs[0] in self.virt_to_meta_dict
            targets_are_virtual = targets[0] in self.virt_to_meta_dict

            if targets_are_virtual and not srcs_are_virtual:
                active_multi_commodity_srcs_dict[mc_id] = tuple(commod_list)
            elif srcs_are_virtual and not targets_are_virtual:
                active_multi_commodity_targets_dict[mc_id] = tuple(commod_list)

            for k in commod_list:
                if k in self.commod_id_to_meta_commod_id:
                    k_meta = self.commod_id_to_meta_commod_id[k]
                    meta_commod_key = self.meta_commodity_list[k_meta]
                    if k_meta not in active_meta_commodity_dict:
                        active_meta_commodity_dict[k_meta] = meta_commod_key
        if self.VERBOSE:
            self._print("--> amcd: ", active_meta_commodity_dict)

        for var in model.getVars():
            if not var.varName.startswith("fp") or var.x <= EPS:
                continue
            match = re.match(r"fp(\d+)_mc(\d+)", var.varName)
            p, mc = int(match.group(1)), int(match.group(2))

            mc_id_to_path_id_to_flow[mc][p] = var.x

            srcs, targets, total_demand, commod_ids = multi_commodity_list[mc]
            srcs_are_virtual = srcs[0] in self.virt_to_meta_dict
            if self.DEBUG:
                for src in srcs[1:]:
                    assert srcs_are_virtual == (src in self.virt_to_meta_dict)

            targets_are_virtual = targets[0] in self.virt_to_meta_dict
            if self.DEBUG:
                for target in targets[1:]:
                    assert targets_are_virtual == (target in self.virt_to_meta_dict)

            if srcs_are_virtual or targets_are_virtual:
                # transit, leavers, or enders
                k_meta = self.commod_id_to_meta_commod_id[commod_ids[0]]
                if self.DEBUG:
                    for k in commod_ids[1:]:
                        assert self.commod_id_to_meta_commod_id[k] == k_meta
                meta_commod_key = self.meta_commodity_list[k_meta]

                meta_sol_dict_def[meta_commod_key] += [
                    (edge, var.x) for edge in path_to_edge_list(all_paths[p])
                ]
            else:
                # purely local flow
                if self.DEBUG:
                    assert len(srcs) == 1 and len(targets) == 1 and len(commod_ids) == 1
                commod_key = (commod_ids[0], (srcs[0], targets[0], total_demand))
                intra_sol_dict_def[commod_key] += [
                    (edge, var.x) for edge in path_to_edge_list(all_paths[p])
                ]

            # more book-keeping for additional reconciliation
            if srcs_are_virtual and not targets_are_virtual:
                r2_targets_in_flow_lists_def[tuple(commod_ids)] += [
                    (edge, var.x) for edge in path_to_edge_list(all_paths[p])
                ]
                k_meta = self.commod_id_to_meta_commod_id[commod_ids[0]]
                assert len(targets) == 1
                target = targets[0]
                other_meta_node = self.meta_commodity_list[k_meta][1][0]
                self.r2_total_flow_in[target][other_meta_node] += var.x

            if targets_are_virtual and not srcs_are_virtual:
                r2_srcs_out_flow_lists_def[tuple(commod_ids)] += [
                    (edge, var.x) for edge in path_to_edge_list(all_paths[p])
                ]
                k_meta = self.commod_id_to_meta_commod_id[commod_ids[0]]
                assert len(srcs) == 1
                source = srcs[0]
                other_meta_node = self.meta_commodity_list[k_meta][1][1]
                self.r2_total_flow_out[source][other_meta_node] += var.x

        if self.VERBOSE:
            self._print("r2_total_flow_in=", self.r2_total_flow_in)
            self._print("r2 total_flow_out=", self.r2_total_flow_out)

        for commod_ids, flow_list in self._create_sol_dict(
            r2_srcs_out_flow_lists_def, list(active_multi_commodity_srcs_dict.values())
        ).items():
            self.r2_srcs_out_flow_lists[commod_ids] = flow_list

        for commod_ids, flow_list in self._create_sol_dict(
            r2_targets_in_flow_lists_def,
            list(active_multi_commodity_targets_dict.values()),
        ).items():
            self.r2_targets_in_flow_lists[commod_ids] = flow_list

        # Set zero-flow commodities to be empty lists
        return (
            self._create_sol_dict(intra_sol_dict_def, intra_commodity_list),
            self._create_sol_dict(
                meta_sol_dict_def, list(active_meta_commodity_dict.values())
            ),
            mc_id_to_path_id_to_flow,
        )

    def extract_r2_sol_as_mat(self, model, G, num_commodities, all_paths):
        edge_idx = {edge: e for e, edge in enumerate(G.edges)}
        sol_mat = np.zeros((len(edge_idx), num_commodities), dtype=np.float32)
        for var in model.getVars():
            if var.varName.startswith("fp") and var.x > EPS:
                match = re.match(r"fp(\d+)_mc(\d+)", var.varName)
                p, mc = int(match.group(1)), int(match.group(2))
                for edge in path_to_edge_list(all_paths[p]):
                    sol_mat[edge_idx[edge], mc] += var.x

        return sol_mat

    def extract_sol_as_dict(
        self, model, commodity_list, path_id_to_commod_id, all_paths
    ):
        sol_dict_def = defaultdict(list)
        for var in model.getVars():
            if var.varName.startswith("f[") and var.x > EPS:
                match = re.match(r"f\[(\d+)\]", var.varName)
                p = int(match.group(1))
                k, (s_k, t_k, d_k) = commodity_list[path_id_to_commod_id[p]]

                sol_dict_def[(k, (s_k, t_k, d_k))] += [
                    (edge, var.x) for edge in path_to_edge_list(all_paths[p])
                ]

        return self._create_sol_dict(sol_dict_def, commodity_list)

    def extract_sol_by_paths(self, model, commodity_list, path_id_to_commod_id):
        sol_paths_def = defaultdict(dict)
        for var in model.getVars():
            if var.varName.startswith("f[") and var.x > EPS:
                match = re.match(r"f\[(\d+)\]", var.varName)
                path_id = int(match.group(1))
                k, (s_k, t_k, d_k) = commodity_list[path_id_to_commod_id[path_id]]
                if k not in sol_paths_def:
                    sol_paths_def[k] = {}
                sol_paths_def[k][path_id] = var.x

        return sol_paths_def

    def extract_sol_as_mat(self, model, G, path_id_to_commod_id, all_paths):
        edge_idx = {edge: e for e, edge in enumerate(G.edges)}
        sol_mat = np.zeros(
            (len(edge_idx), len(set(path_id_to_commod_id.values()))), dtype=np.float32
        )
        for var in model.getVars():
            if var.varName.startswith("f[") and var.x > EPS:
                match = re.match(r"f\[(\d+)\]", var.varName)
                p = int(match.group(1))
                k = path_id_to_commod_id[p]
                for edge in path_to_edge_list(all_paths[p]):
                    sol_mat[edge_idx[edge], k] += var.x

        return sol_mat

    ##################
    # LP FORMULATION #
    ##################

    def _define_max_util_obj(self, m, path_vars, commodities, GAMMA=1e-3):
        obj = quicksum(path_vars)
        if self.VERBOSE:
            self._print("GAMMA for path: {}".format(GAMMA))
        max_util_vars = m.addVars(
            len(commodities), vtype=GRB.CONTINUOUS, lb=0.0, name="z"
        )
        m.update()
        for k, rest in enumerate(commodities):
            path_ids = rest[-1]
            for p in path_ids:
                m.addConstr(path_vars[p] <= max_util_vars[k])
        obj -= GAMMA * quicksum(max_util_vars)
        return obj

    def _r1_lp(self, paths_dict, meta_commodity_list):
        if self.VERBOSE:
            self._print(
                "--> R1 lp: #n =",
                len(self.G_meta.nodes),
                " #e=",
                len(self.G_meta.edges),
                " #c=",
                len(meta_commodity_list),
                " total demand=",
                sum([d_k for (_, (_, _, d_k)) in meta_commodity_list]),
            )

        commodities = []
        edge_to_paths = defaultdict(list)
        # node id to list of r1 paths through it
        node_to_paths = defaultdict(list)
        self._r1_paths = []
        self._r1_path_to_commod = {}
        self._r1_commod_to_path_ids = defaultdict(list)

        num_paths = []
        path_i = 0
        for k, (s_k, t_k, d_k) in meta_commodity_list:
            paths = paths_dict[(s_k, t_k)]
            path_ids = []
            assert len(paths) == 1
            path = paths[0]

            self._r1_paths.append(path)

            for edge in path_to_edge_list(path):
                edge_to_paths[edge].append(path_i)
            for n in path:
                node_to_paths[n].append(path_i)
            path_ids.append(path_i)

            self._r1_path_to_commod[path_i] = k
            self._r1_commod_to_path_ids[k].append(path_i)
            path_i += 1

            commodities.append((k, d_k, path_ids))

            num_paths.append(len(path_ids))

        if self.VERBOSE:
            self._print(
                "--> R1 #total paths=",
                sum(num_paths),
                " avg #paths per commodity=",
                np.mean(num_paths),
            )
        self._r1_commod_to_path_ids = dict(self._r1_commod_to_path_ids)
        edge_to_paths = dict(edge_to_paths)

        if self.DEBUG:
            assert len(self._r1_paths) == path_i

        m = Model("max-flow: R1")

        # Create variables: one for each path
        path_vars = m.addVars(path_i, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

        # Set objective
        if self.VERBOSE:
            self._print("Not applying min max util in R1")
        obj = quicksum(path_vars)

        m.setObjective(obj, GRB.MAXIMIZE)

        # Add demand constraints
        for _, d_k, path_ids in commodities:
            m.addConstr(quicksum(path_vars[p] for p in path_ids) <= d_k)

        # Add edge capacity constraints
        for u, v, c_e in self.G_meta.edges.data("capacity"):
            if (u, v) in edge_to_paths:
                paths = edge_to_paths[(u, v)]
                constr_vars = [path_vars[p] for p in paths]
                m.addConstr(quicksum(constr_vars) <= c_e)

        return LpSolver(m, None, self.DEBUG, self.VERBOSE, self.out)

    def _r2_lp(
        self,
        curr_meta_node,
        paths_dict,
        G_hat,
        all_v_hat_in,
        all_v_hat_out,
        intra_commods,
        min_max_util=True,
    ):
        debug_r2 = False

        partition_vector = self._partition_vector

        subgraph_nodes = np.argwhere(partition_vector == curr_meta_node).flatten()
        if self.VERBOSE or debug_r2:
            self._print(
                "--> R2 M",
                curr_meta_node,
                " #n= ",
                len(G_hat.nodes),
                " #e= ",
                len(G_hat.edges),
            )
            self._print("--> nodes: ", sorted(subgraph_nodes))

        # 2) commodities
        multi_commodity_list = []
        meta_commod_to_multi_commod_ids = defaultdict(list)

        # commods local to this meta-node
        for k, (s_k, t_k, d_k) in intra_commods:
            multi_commodity_list.append(([s_k], [t_k], d_k, [k]))

        if self.VERBOSE or debug_r2:
            self._print(
                "--># intra commods (in multi_commodity_list)=",
                len(multi_commodity_list),
            )

        for meta_commod_key, flow_seq in self.r1_sol_dict.items():
            if len(flow_seq) == 0:
                continue
            (k_meta, (s_k_meta, t_k_meta, _)) = meta_commod_key
            orig_commod_list_in_k_meta = self.meta_commodity_dict[meta_commod_key]

            # leavers
            if s_k_meta == curr_meta_node:
                for u in subgraph_nodes:
                    commod_ids = [
                        k for k, (s_k, _, d_k) in orig_commod_list_in_k_meta if s_k == u
                    ]
                    if len(commod_ids) > 0:
                        total_demand = sum(
                            [
                                d_k
                                for _, (s_k, _, d_k) in orig_commod_list_in_k_meta
                                if s_k == u
                            ]
                        )

                        targets = list(all_v_hat_out)

                        meta_commod_to_multi_commod_ids[k_meta].append(
                            len(multi_commodity_list)
                        )
                        multi_commodity_list.append(
                            ([u], targets, total_demand, commod_ids)
                        )

            # incomers
            elif t_k_meta == curr_meta_node:
                for v in subgraph_nodes:
                    commod_ids = [
                        k for k, (_, t_k, _) in orig_commod_list_in_k_meta if t_k == v
                    ]
                    if len(commod_ids) > 0:
                        total_demand = sum(
                            [
                                d_k
                                for _, (_, t_k, d_k) in orig_commod_list_in_k_meta
                                if t_k == v
                            ]
                        )

                        sources = list(all_v_hat_in)

                        meta_commod_to_multi_commod_ids[k_meta].append(
                            len(multi_commodity_list)
                        )

                        multi_commodity_list.append(
                            (sources, [v], total_demand, commod_ids)
                        )

            # transit
            else:
                meta_in, meta_out = get_in_and_out_neighbors(flow_seq, curr_meta_node)
                assert (len(meta_in) > 0 and len(meta_out) > 0) or (
                    len(meta_in) == 0 and len(meta_out) == 0
                )

                if len(meta_in) == 0:
                    continue

                commod_ids = [k for k, _ in orig_commod_list_in_k_meta]
                total_demand = sum(
                    [d_k for _, (_, _, d_k) in orig_commod_list_in_k_meta]
                )
                meta_commod_to_multi_commod_ids[k_meta].append(
                    len(multi_commodity_list)
                )
                multi_commodity_list.append(
                    (
                        [self.meta_to_virt_dict[u][0] for u in meta_in],
                        [self.meta_to_virt_dict[u][1] for u in meta_out],
                        total_demand,
                        commod_ids,
                    )
                )

        if self.VERBOSE or debug_r2:
            self._print("--> multi commodity list: ", multi_commodity_list)

        if debug_r2:
            print("mk88 = mcls{}".format(meta_commod_to_multi_commod_ids[88]))
            for mcl_id in meta_commod_to_multi_commod_ids[88]:
                print("mcl{} = {}".format(mcl_id, multi_commodity_list[mcl_id]))

        if len(multi_commodity_list) == 0:
            return None, multi_commodity_list, None, None

        # 3) get the paths set up
        all_paths = []  # a list of all paths
        next_path_id = 0
        # (source, target) -> ([path ids])
        # (s_k, t_k) -> [path ids]; used to avoid duplicate path computation
        source_target_paths = defaultdict(list)
        v_hat_in_paths = defaultdict(list)  # source -> ([path ids])
        v_hat_out_paths = defaultdict(list)  # targret -> ([path ids])
        path_id_to_multi_commod_ids = (
            []
        )  # path id -> [commod ids in multi_commodity_list]
        edge_to_path_ids = defaultdict(list)  # edge -> [path ids]

        if self.VERBOSE:
            self._print("M{", curr_meta_node, "}; mcl={", multi_commodity_list, "}")
        for k, (s_k_list, t_k_list, _, k_list) in enumerate(multi_commodity_list):
            if debug_r2:
                print(
                    "MCL{} k_meta={} s{} -> t{}".format(
                        k,
                        self.commod_id_to_meta_commod_id[k_list[0]]
                        if k_list[0] in self.commod_id_to_meta_commod_id
                        else "",
                        s_k_list,
                        t_k_list,
                    )
                )
            for s_k, t_k in product(s_k_list, t_k_list):
                if (s_k, t_k) not in source_target_paths:
                    new_paths = paths_dict[(s_k, t_k)]
                    for path in new_paths:
                        all_paths.append(path)
                        path_id_to_multi_commod_ids.append([])
                        source_target_paths[(s_k, t_k)].append(next_path_id)
                        if s_k in all_v_hat_in:
                            v_hat_in_paths[s_k].append(next_path_id)
                        if t_k in all_v_hat_out:
                            v_hat_out_paths[t_k].append(next_path_id)
                        for edge in path_to_edge_list(path):
                            edge_to_path_ids[edge].append(next_path_id)

                        next_path_id += 1

                for path_id in source_target_paths[(s_k, t_k)]:
                    path = all_paths[path_id]
                    path_id_to_multi_commod_ids[path_id].append(k)

        if self.DEBUG:
            assert len(all_paths) == len(path_id_to_multi_commod_ids)

        edge_to_path_ids = dict(edge_to_path_ids)
        v_hat_in_paths = dict(v_hat_in_paths)
        v_hat_out_paths = dict(v_hat_out_paths)

        if self.VERBOSE or debug_r2:
            self._print("--> all_paths:", all_paths)
            self._print("--> path_id_to_multi_commod_ids:", path_id_to_multi_commod_ids)

        m = Model("max-flow: R2, metanode {}".format(curr_meta_node))
        path_id_to_commod_id_to_var = defaultdict(dict)
        mc_id_to_path_id_to_var = defaultdict(dict)
        mc_id_to_path_ids = defaultdict(list)
        if self.VERBOSE or debug_r2:
            seen_var_ids = set()
        all_vars = []
        for path_id, multi_commod_ids in enumerate(path_id_to_multi_commod_ids):
            for multi_commod_id in multi_commod_ids:
                gb_var = m.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name="fp{}_mc{}".format(path_id, multi_commod_id),
                )

                if self.VERBOSE or debug_r2:
                    assert (path_id, multi_commod_id) not in seen_var_ids
                    seen_var_ids.add((path_id, multi_commod_id))

                path_id_to_commod_id_to_var[path_id][multi_commod_id] = gb_var
                mc_id_to_path_ids[multi_commod_id].append(path_id)
                mc_id_to_path_id_to_var[multi_commod_id][path_id] = gb_var
                all_vars.append(gb_var)
        m.update()

        # Set objective
        if min_max_util:
            if self.VERBOSE or debug_r2:
                self._print("Applying min max util in R2")
            obj = quicksum(all_vars)
            GAMMA = 1e-2 / max(
                1.0, max([demand for _, _, demand, _ in multi_commodity_list])
            )
            if self.VERBOSE or debug_r2:
                self._print("GAMMA for path: {}".format(GAMMA))

            for k in mc_id_to_path_id_to_var.keys():
                max_path_var = m.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, name="maxp_mcid{}".format(k)
                )
                for var in mc_id_to_path_id_to_var[k].values():
                    m.addConstr(var <= max_path_var)
                obj -= GAMMA * max_path_var
            m.update()
        else:
            if self.VERBOSE or debug_r2:
                self._print("Not applying min max util in R2")
            obj = quicksum(all_vars)

        m.setObjective(obj, GRB.MAXIMIZE)

        # Add demand constraints
        for multi_commod_id, (_, _, demand, _) in enumerate(multi_commodity_list):
            m.addConstr(
                quicksum(mc_id_to_path_id_to_var[multi_commod_id].values()) <= demand
            )

        # Add edge capacity constraints
        for u, v, c_e in G_hat.edges.data("capacity"):
            if (u, v) in edge_to_path_ids:
                path_ids = edge_to_path_ids[(u, v)]
                constr_vars = [
                    var
                    for p in path_ids
                    for var in path_id_to_commod_id_to_var[p].values()
                ]
                m.addConstr(quicksum(constr_vars) <= c_e)

        # Add meta-flow constraints
        meta_edge_inds = {
            edge: e
            for e, edge in enumerate(self.G_meta.edges())
            if edge[0] == curr_meta_node or edge[-1] == curr_meta_node
        }

        for k_meta, multi_commod_ids_list in meta_commod_to_multi_commod_ids.items():
            if self.VERBOSE or debug_r2:
                self._print(
                    "--> k_meta:", k_meta, "; multi commod ids:", multi_commod_ids_list
                )
            s_k_meta, t_k_meta, _ = self.meta_commodity_list[k_meta][-1]
            if s_k_meta != curr_meta_node:
                for v_hat_in in all_v_hat_in:
                    v_meta = self.virt_to_meta_dict[v_hat_in]
                    meta_in_flow = self.r1_sol_mat[
                        meta_edge_inds[(v_meta, curr_meta_node)], k_meta
                    ]
                    if (
                        v_hat_in not in v_hat_in_paths
                        or len(v_hat_in_paths[v_hat_in]) == 0
                    ):
                        if meta_in_flow > 0.001:
                            print(
                                "WARN: v_hat_in{} with flow{} has no grb vars".format(
                                    v_hat_in, meta_in_flow
                                )
                            )
                    else:
                        constr_vars = [
                            path_id_to_commod_id_to_var[p][multi_commod_id]
                            for p in v_hat_in_paths[v_hat_in]
                            for multi_commod_id in path_id_to_multi_commod_ids[p]
                            if multi_commod_id in multi_commod_ids_list
                        ]
                        m.addConstr(quicksum(constr_vars) <= meta_in_flow)
            if t_k_meta != curr_meta_node:
                for v_hat_out in all_v_hat_out:
                    v_meta = self.virt_to_meta_dict[v_hat_out]
                    meta_out_flow = self.r1_sol_mat[
                        meta_edge_inds[(curr_meta_node, v_meta)], k_meta
                    ]
                    if (
                        v_hat_out not in v_hat_out_paths
                        or len(v_hat_out_paths[v_hat_out]) == 0
                    ):
                        if meta_out_flow > 0.001:
                            print(
                                "WARN: v_hat_out{} with flow{} has now grb vars".format(
                                    v_hat_out, meta_out_flow
                                )
                            )
                    else:
                        constr_vars = [
                            path_id_to_commod_id_to_var[p][multi_commod_id]
                            for p in v_hat_out_paths[v_hat_out]
                            for multi_commod_id in path_id_to_multi_commod_ids[p]
                            if multi_commod_id in multi_commod_ids_list
                        ]
                        m.addConstr(quicksum(constr_vars) <= meta_out_flow)

        if self.VERBOSE or debug_r2:
            model_output_file = "r2_m" + str(curr_meta_node) + ".lp"
            m.write(model_output_file)
            self._print("--> r2 model written to: ", model_output_file)

        return (
            LpSolver(m, None, self.DEBUG, self.VERBOSE, self.out),
            multi_commodity_list,
            all_paths,
            dict(mc_id_to_path_ids),
        )

    def _r3_lp(self, meta_commodities, constrain_r3_by_r1=True):
        debug_r3 = False

        G = self.G_meta
        commodity_list = meta_commodities
        all_paths = self._r1_paths
        self._r3_path_to_commod = {}
        self._r3_paths = []
        r3_path_id_to_r1_path_id = {}

        commodities = []
        edge_to_paths = defaultdict(list)
        path_i = 0
        for r3_k, (r1_k, (s_k, t_k, d_k)) in enumerate(commodity_list):
            paths = [(p_i, all_paths[p_i]) for p_i in self._r1_commod_to_path_ids[r1_k]]
            path_ids = []
            for r1_path_id, path in paths:
                for edge in path_to_edge_list(path):
                    edge_to_paths[edge].append(path_i)
                path_ids.append(path_i)
                r3_path_id_to_r1_path_id[path_i] = r1_path_id

                self._r3_paths.append(path)
                self._r3_path_to_commod[path_i] = r3_k
                path_i += 1
            commodities.append((r1_k, s_k, t_k, d_k, path_ids))

        edge_to_paths = dict(edge_to_paths)

        if self.DEBUG:
            assert len(all_paths) == path_i

        m = Model("max-flow: R3")
        # Create variables: one for each path
        path_vars = m.addVars(path_i, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

        # Set objective
        obj = quicksum(path_vars)
        if self.VERBOSE:
            self._print("Not applying min max util in R3")
        obj = quicksum(path_vars)

        m.setObjective(obj, GRB.MAXIMIZE)

        if self.VERBOSE:
            self._print("#paths =", path_i)
            self._print(commodities)

        # Add demand constraints
        for k_meta, s_k, t_k, d_k, path_ids in commodities:
            if self.VERBOSE:
                self._print("doing meta: ", k_meta)
            m.addConstr(quicksum(path_vars[p] for p in path_ids) <= d_k)

            # Add edge commod cap constraints per meta-edge
            vars_per_meta_edge = defaultdict(list)
            for r3_path_id in path_ids:
                for u_meta, v_meta in path_to_edge_list(self._r3_paths[r3_path_id]):
                    vars_per_meta_edge[(u_meta, v_meta)].append(path_vars[r3_path_id])

            meta_commod_key = self.meta_commodity_list[k_meta]
            for (u_meta, v_meta), edge_vars in vars_per_meta_edge.items():
                recon_flow = 0.0
                if meta_commod_key in self.reconciliation_sol_dicts[(u_meta, v_meta)]:
                    recon_flow_list = self.reconciliation_sol_dicts[(u_meta, v_meta)][
                        meta_commod_key
                    ]
                    for (u, v), l in recon_flow_list:
                        # this is in case we add more edges to reconciliation at some point
                        if (
                            self._partition_vector[u] == u_meta
                            and self._partition_vector[v] == v_meta
                        ):
                            recon_flow += l
                    if self.VERBOSE:
                        self._print(
                            "--> edge: ({}, {}), flow: {}".format(
                                u_meta, v_meta, recon_flow
                            )
                        )
                m.addConstr(quicksum(edge_vars) <= recon_flow)

        # Add edge capacity constraints
        for u, v, c_e in G.edges.data("capacity"):
            if (u, v) in edge_to_paths:
                paths = edge_to_paths[(u, v)]
                constr_vars = [path_vars[p] for p in paths]
                m.addConstr(quicksum(constr_vars) <= c_e)

        if self.VERBOSE or debug_r3:
            model_output_file = "r3.lp"
            m.write(model_output_file)
            self._print("--> r3 model written to: ", model_output_file)

        return LpSolver(m, None, self.DEBUG, self.VERBOSE, self.out)

    ########################
    # BEGIN RECONCILIATION #
    ########################

    def _reconciliation_lp(self, u_meta, v_meta):
        G_u_meta_v_meta = nx.DiGraph()

        debug_recon = False

        # 1) get nodes that originate in u_meta and connect to v_meta
        candidate_u_meta_nodes = np.argwhere(u_meta == self._partition_vector).flatten()
        nodes_in_u_meta, nodes_in_v_meta = set(), set()

        for u in candidate_u_meta_nodes:
            for v in self.problem.G.successors(u):
                if self._partition_vector[v] == v_meta:
                    if u not in nodes_in_u_meta:
                        nodes_in_u_meta.add(u)
                        G_u_meta_v_meta.add_node(u)
                    if v not in nodes_in_v_meta:
                        nodes_in_v_meta.add(v)
                        G_u_meta_v_meta.add_node(v)
                    G_u_meta_v_meta.add_edge(
                        u, v, capacity=self.problem.G[u][v]["capacity"]
                    )

        edges_list = list(G_u_meta_v_meta.edges.data("capacity"))
        edge_idx = {(edge[0], edge[1]): e for e, edge in enumerate(edges_list)}
        print("edges in recon: {}".format(edge_idx))

        # Only use the meta_sol_dicts from R2
        u_meta_sol_dict, v_meta_sol_dict = (
            self.r2_meta_sols_dicts[u_meta],
            self.r2_meta_sols_dicts[v_meta],
        )
        u_hat_in, _ = self.meta_to_virt_dict[u_meta]
        _, v_hat_out = self.meta_to_virt_dict[v_meta]

        common_meta_commods = list(
            set(u_meta_sol_dict.keys()).intersection(set(v_meta_sol_dict.keys()))
        )

        # print("--> cmc: ", len(common_meta_commods), ' ', common_meta_commods)

        # Filter out meta commodities that aren't flowing out of u_meta
        common_meta_commods = [
            meta_commod_key
            for meta_commod_key in common_meta_commods
            if len(
                neighbors_and_flows(u_meta_sol_dict[meta_commod_key], -1, {v_hat_out})
            )
            > 0
        ]
        # print("--> cmc: ", len(common_meta_commods), ' ', common_meta_commods)

        common_meta_commods = [
            meta_commod_key
            for meta_commod_key in common_meta_commods
            if len(neighbors_and_flows(v_meta_sol_dict[meta_commod_key], 0, {u_hat_in}))
            > 0
        ]
        # print("--> cmc: ", len(common_meta_commods), ' ', common_meta_commods)

        # 2) Construct model
        m = Model(
            "Reconciliation, meta-nodes {} (out) and {} (in)".format(u_meta, v_meta)
        )

        commod_vars = m.addVars(
            len(edges_list),
            len(common_meta_commods),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="f",
        )

        tot_u_flow = dict()
        tot_v_flow = dict()

        # Flow constraints from R2
        for k, meta_commod_key in enumerate(common_meta_commods):
            u_meta_flow_list = u_meta_sol_dict[meta_commod_key]
            v_meta_flow_list = v_meta_sol_dict[meta_commod_key]
            if self.VERBOSE:
                self._print("k_local{}= mk{}".format(k, meta_commod_key))
                self._print("u meta flow list: {}".format(u_meta_flow_list))
                self._print("v meta flow list: {}".format(v_meta_flow_list))
                self._print()

            u_meta_src_outflow_dict = defaultdict(float)
            v_meta_sink_inflow_dict = defaultdict(float)

            k_meta, _ = meta_commod_key

            tot_u_flow[k_meta] = 0
            tot_v_flow[k_meta] = 0

            # For each node u_src in u_meta, we examine the flow f_k that
            # needs to be pushed out of it (i.e., the outflow that it sent in
            # R2). For the edges that are coming out of u_src, the sum of those
            # flow-edge variables should be equal to f_k
            for u_src, out_flow in neighbors_and_flows(
                u_meta_flow_list, -1, {v_hat_out}
            ):
                u_meta_src_outflow_dict[u_src] += out_flow

            for u_src in nodes_in_u_meta:
                outflow = u_meta_src_outflow_dict[u_src]
                if outflow <= 0.0:
                    outflow = 0.0

                outgoing_edge_idxs = [
                    edge_idx[(u_src, v)] for v in G_u_meta_v_meta.successors(u_src)
                ]
                m.addConstr(
                    quicksum(commod_vars[e, k] for e in outgoing_edge_idxs) <= outflow,
                    "outflow_u{}_mk{}".format(u_src, k_meta),
                )

                tot_u_flow[k_meta] += outflow

            # We do the same for every v_sink in v_meta: we examine the flow
            # f_k that needs to be pushed *into* it (i.e., the inflow that
            # it received in R2). For the edges that are coming into v_sink,
            # the sum of those flow-edge variables should be equal to f_k
            for v_sink, in_flow in neighbors_and_flows(v_meta_flow_list, 0, {u_hat_in}):
                v_meta_sink_inflow_dict[v_sink] += in_flow

            for v_sink in nodes_in_v_meta:
                inflow = v_meta_sink_inflow_dict[v_sink]
                if inflow <= 0.0:
                    inflow = 0.0

                incoming_edge_idxs = [
                    edge_idx[(u, v_sink)] for u in G_u_meta_v_meta.predecessors(v_sink)
                ]

                m.addConstr(
                    quicksum(commod_vars[e, k] for e in incoming_edge_idxs) <= inflow,
                    "inflow_v{}_mk{}".format(v_sink, k_meta),
                )

                tot_v_flow[k_meta] += inflow

        # edge capacity constraints
        for e, (u, v, c_e) in enumerate(edges_list):
            m.addConstr(
                commod_vars.sum(e, "*") <= c_e, "cap_e{}_u{}_v{}".format(e, u, v)
            )

        # Set objective: maximize total flow
        obj = quicksum(commod_vars)
        m.setObjective(obj, GRB.MAXIMIZE)

        if self.VERBOSE:
            self._print("--> total_u_flow=", sum(tot_u_flow.values()), "; ", tot_u_flow)
            self._print("--> total_v_flow=", sum(tot_v_flow.values()), "; ", tot_v_flow)

        if debug_recon:
            model_output_file = "recon_%sout_%sin.lp" % (u_meta, v_meta)
            m.write(model_output_file)
            self._print("--> recon model written to: ", model_output_file)

        for k_meta in tot_u_flow.keys():
            self.before_recon_meta_out_flow[k_meta][(u_meta, v_meta)] = tot_u_flow[
                k_meta
            ]
            self.before_recon_meta_in_flow[k_meta][(u_meta, v_meta)] = tot_v_flow[
                k_meta
            ]

        return (
            LpSolver(m, None, self.DEBUG, self.VERBOSE, self.out),
            G_u_meta_v_meta,
            common_meta_commods,
            nodes_in_u_meta,
            nodes_in_v_meta,
        )

    ######################
    # END RECONCILIATION #
    ######################

    ###################
    # BEGIN KIRCHOFFS #
    ###################
    def _kirchoffs_lp(self, meta_commod_key, commodity_list):
        all_commod_ids = [commod_key[0] for commod_key in commodity_list]
        commod_id_to_ind = {k: i for i, k in enumerate(all_commod_ids)}
        u_meta, v_meta = meta_commod_key[-1][0], meta_commod_key[-1][1]

        m = Model(
            "Kirchoff's Law, meta-nodes {} (out) and {} (in)".format(u_meta, v_meta)
        )

        commod_vars = m.addVars(
            len(all_commod_ids), vtype=GRB.CONTINUOUS, lb=0.0, name="f"
        )

        obj = quicksum(commod_vars)
        m.setObjective(obj, GRB.MAXIMIZE)

        # Demand constraints
        for k, (_, _, d_k) in commodity_list:
            if self.VERBOSE:
                self._print(k, d_k)
            m.addConstr(commod_vars[commod_id_to_ind[k]] <= d_k)

        # Kirchoff constraints
        src_out_flows = self.r2_src_out_flows[(u_meta, v_meta)]
        for commod_ids, total_flow in src_out_flows.items():
            if self.VERBOSE:
                self._print(commod_ids, total_flow)
            if total_flow <= 0.0:
                total_flow = 0.0
            m.addConstr(
                quicksum([commod_vars[commod_id_to_ind[k]] for k in commod_ids])
                <= total_flow
            )

        target_in_flows = self.r2_target_in_flows[(u_meta, v_meta)]
        for commod_ids, total_flow in target_in_flows.items():
            if self.VERBOSE:
                self._print(commod_ids, total_flow)
            if total_flow <= 0.0:
                total_flow = 0.0
            m.addConstr(
                quicksum([commod_vars[commod_id_to_ind[k]] for k in commod_ids])
                <= total_flow
            )

        return LpSolver(m, None, self.DEBUG, self.VERBOSE, self.out)

    def _extract_kirchoffs_sol(self, model, commodity_list):
        flow_dict = {}
        for var in model.getVars():
            match = re.match(r"f\[(\d+)\]", var.varName)
            i = int(match.group(1))
            k, _ = commodity_list[i]
            if var.x < EPS:
                flow_dict[k] = 0.0
            else:
                flow_dict[k] = var.x

        return flow_dict

    def divide_into_multi_commod_flows(
        self, multi_commod_flow_lists, src_or_target_idx
    ):
        multi_commod_flows_per_k_meta = defaultdict(dict)

        for commod_ids, flow_list in multi_commod_flow_lists.items():
            k = commod_ids[0]
            rest = self.problem.commodity_list[k][-1]
            src_or_target = rest[src_or_target_idx]
            src, target = rest[0], rest[1]
            u_meta, v_meta = self._partition_vector[src], self._partition_vector[target]

            if self.DEBUG:
                for k in commod_ids[1:]:
                    rest = self.problem.commodity_list[k][-1]
                    src, target = rest[0], rest[1]
                    assert src_or_target == rest[src_or_target_idx]
                    assert u_meta == self._partition_vector[src]
                    assert v_meta == self._partition_vector[target]

            multi_commod_flows_per_k_meta[(u_meta, v_meta)][
                commod_ids
            ] = compute_in_or_out_flow(flow_list, src_or_target_idx, {src_or_target})
        return dict(multi_commod_flows_per_k_meta)

    #################
    # END KIRCHOFFS #
    #################

    def solve(
        self,
        problem,
        partition_vector,
        G_meta,
        r1_paths_dict,
        r2_paths_dicts,
        r2_G_hats,
        all_v_hat_ins,
        all_v_hat_outs,
        intra_commods_lists,
        meta_commodity_list,
        meta_commodity_dict,
        commod_id_to_meta_commod_id,
        virt_to_meta_dict,
        meta_to_virt_dict,
        selected_inter_edges,
        r1_method=Method.BARRIER,
        r2_method=Method.PRIMAL_SIMPLEX,
        reconciliation_method=Method.CONCURRENT,
        r3_method=Method.CONCURRENT,
    ):

        self._problem = problem
        self._partition_vector = partition_vector
        self.G_meta = G_meta
        self.meta_commodity_list = meta_commodity_list
        self.meta_commodity_dict = meta_commodity_dict
        self.commod_id_to_meta_commod_id = commod_id_to_meta_commod_id
        self.virt_to_meta_dict = virt_to_meta_dict
        self.meta_to_virt_dict = meta_to_virt_dict
        self.selected_inter_edges = selected_inter_edges

        # R1
        self._runtime_dict = {}
        if self.VERBOSE:
            self._print("R1")

        r1_solver = self._r1_lp(r1_paths_dict, self.meta_commodity_list)
        r1_solver.gurobi_out = self.out.name.replace(".txt", "-r1.txt")
        r1_solver.solve_lp(r1_method)
        self._runtime_dict["r1"] = r1_solver.model.Runtime
        self.r1_obj_val = r1_solver.obj_val
        self.r1_sol_mat = self.extract_sol_as_mat(
            r1_solver.model, self.G_meta, self._r1_path_to_commod, self._r1_paths
        )
        self.r1_sol_dict = self.extract_sol_as_dict(
            r1_solver.model,
            self.meta_commodity_list,
            self._r1_path_to_commod,
            self._r1_paths,
        )

        if self.VERBOSE:
            self._print(self.r1_sol_dict)
        self._save_pkl(
            self.r1_sol_dict, self.out.name.replace(".txt", "-r1-sol-dict.pkl")
        )

        # R2
        if self.VERBOSE:
            self._print("\nR2")
        self._runtime_dict["r2"] = {}
        self.r2_meta_sols_dicts = []
        self.r2_sols_mats = []
        self.intra_sols_dicts = []
        self.intra_obj_vals = [0.0 for _ in self.G_meta.nodes]
        self.r2_models = []
        self.r2_mc_lists = []
        self.r2_paths = []

        # per 'target', the total flow from sources in a given meta-node
        self.r2_total_flow_in = defaultdict(lambda: defaultdict(float))
        # per 'source', the total flow to targets in a given meta-node
        self.r2_total_flow_out = defaultdict(lambda: defaultdict(float))

        self.r2_srcs_out_flow_lists, self.r2_targets_in_flow_lists = {}, {}

        for meta_node_id in self.G_meta.nodes:
            self._print("\nR2, meta-node {}".format(meta_node_id))

            r2_paths_dict = r2_paths_dicts[meta_node_id]
            G_hat = r2_G_hats[meta_node_id]
            all_v_hat_in = all_v_hat_ins[meta_node_id]
            all_v_hat_out = all_v_hat_outs[meta_node_id]
            intra_commods = intra_commods_lists[meta_node_id]

            (
                r2_solver,
                multi_commodity_list,
                r2_all_paths,
                mc_id_to_path_ids,
            ) = self._r2_lp(
                meta_node_id,
                r2_paths_dict,
                G_hat,
                all_v_hat_in,
                all_v_hat_out,
                intra_commods,
                min_max_util=self.r2_min_max_util,
            )

            # if self.VERBOSE:
            self._print(
                "{} nodes, {} edges in R2 subgraph".format(
                    len(G_hat.nodes), len(G_hat.edges)
                )
            )
            if len(multi_commodity_list) > 0:
                r2_solver.gurobi_out = self.out.name.replace(
                    ".txt", "-r2-{}.txt".format(meta_node_id)
                )
                r2_solver.solve_lp(r2_method)
                self.r2_models.append(r2_solver.model)
                self.r2_mc_lists.append(multi_commodity_list)

                self.r2_paths.append(r2_all_paths)
                self._runtime_dict["r2"][meta_node_id] = r2_solver.model.Runtime

                # Once we solve the first group, those flows do not need to be
                # passed to R3; the reconciled flow should be the final
                # solution, and the residual capacity should be passed to R3

                # Extract flows from the R2 LP, divided into 2 groups:
                # 1) the intra flows (as individual commodities)
                # 2) the inter flows (as meta-commodities)
                (
                    intra_sol_dict,
                    meta_sol_dict,
                    mc_id_to_path_id_to_flow,
                ) = self.extract_r2_sol_as_dict(
                    r2_solver.model,
                    multi_commodity_list,
                    intra_commods_lists[meta_node_id],
                    r2_all_paths,
                )
                self.r2_meta_sols_dicts.append(meta_sol_dict)
                self.intra_sols_dicts.append(intra_sol_dict)
                self.intra_obj_vals[meta_node_id] = 0.0
                for commod_key, flow_list in intra_sol_dict.items():
                    self.intra_obj_vals[meta_node_id] += compute_in_or_out_flow(
                        flow_list, 0, {commod_key[-1][0]}
                    )

                r2_sol_mat = self.extract_r2_sol_as_mat(
                    r2_solver.model, G_hat, len(multi_commodity_list), r2_all_paths
                )
                self.r2_sols_mats.append(r2_sol_mat)

                if self.VERBOSE:
                    self._print("Intra sol dict for R2: {}".format(intra_sol_dict))
                    self._print("Meta sol dict for R2: {}".format(meta_sol_dict))
                self._save_pkl(
                    intra_sol_dict,
                    self.out.name.replace(
                        ".txt", "-r2-{}-intra-sol-dict.pkl".format(meta_node_id)
                    ),
                )
                self._save_pkl(
                    meta_sol_dict,
                    self.out.name.replace(
                        ".txt", "-r2-{}-meta-sol-dict.pkl".format(meta_node_id)
                    ),
                )

            else:
                if self.VERBOSE:
                    self._print("Meta node {} has no R2 commodities")
                self.r2_models.append(None)
                self.r2_mc_lists.append([])
                self.r2_paths.append([])
                self._runtime_dict["r2"][meta_node_id] = 0.0
                self.r2_meta_sols_dicts.append({})
                self.intra_sols_dicts.append({})
                self.intra_obj_vals[meta_node_id] = 0.0
                self.r2_sols_mats.append(np.array([]))
                self._save_pkl(
                    {},
                    self.out.name.replace(
                        ".txt", "-r2-{}-intra-sol-dict.pkl".format(meta_node_id)
                    ),
                )
                self._save_pkl(
                    {},
                    self.out.name.replace(
                        ".txt", "-r2-{}-meta-sol-dict.pkl".format(meta_node_id)
                    ),
                )

        # Meta-edge reconciliation: for each pair of meta-nodes, examine the
        # meta-edges in between them and all "non-local" nodes. Solve a new LP
        # for these meta-edges:
        self._print("\nReconciliation\n")

        self.G_u_meta_v_metas = []
        self.reconciliation_sol_dicts = {}
        self._runtime_dict["reconciliation"] = {}

        # meta_commod_id -> meta_node_id -> out/in flow
        reconciliation_meta_out_flows = defaultdict(lambda: defaultdict(float))
        reconciliation_meta_in_flows = defaultdict(lambda: defaultdict(float))

        # *before* reconciliation; meta_commod_id, -> (u, v) -> flow; result of R2
        self.before_recon_meta_out_flow = defaultdict(lambda: defaultdict(float))
        self.before_recon_meta_in_flow = defaultdict(lambda: defaultdict(float))

        for u_meta, v_meta in self.G_meta.edges:
            if self.VERBOSE:
                self._print(
                    "\nReconciliation: {} (out) and {} (in)".format(u_meta, v_meta)
                )
            (
                reconciliation_solver,
                G_u_meta_v_meta,
                meta_commod_u_out_v_in,
                nodes_in_u_meta,
                nodes_in_v_meta,
            ) = self._reconciliation_lp(u_meta, v_meta)

            if self.VERBOSE or True:
                self._print(
                    "{} nodes, {} edges in reconciliation subgraph".format(
                        len(G_u_meta_v_meta.nodes), len(G_u_meta_v_meta.edges)
                    )
                )
            reconciliation_solver.gurobi_out = self.out.name.replace(
                ".txt", "-reconciliation-{}-{}.txt".format(u_meta, v_meta)
            )
            reconciliation_solver.solve_lp(reconciliation_method)
            self._runtime_dict["reconciliation"][
                (u_meta, v_meta)
            ] = reconciliation_solver.model.Runtime
            self.G_u_meta_v_metas.append(G_u_meta_v_meta)

            # Extract reconciliation solution
            reconciliation_sol_dict = self.extract_reconciliation_sol_as_dict(
                reconciliation_solver.model,
                meta_commod_u_out_v_in,
                list(G_u_meta_v_meta.edges),
            )

            if self.VERBOSE:
                self._print(meta_commod_u_out_v_in)
                self._print(
                    "Reconciliation sol dict: {}".format(reconciliation_sol_dict)
                )

            self.reconciliation_sol_dicts[(u_meta, v_meta)] = reconciliation_sol_dict

            # k_meta to flow
            after_recon_flow = dict()
            for meta_commod, flow_list in reconciliation_sol_dict.items():
                total_flow = 0.0
                for (u, v), l in flow_list:
                    # this is in case we add more edges to reconciliation at some point
                    if (
                        self._partition_vector[u] == u_meta
                        and self._partition_vector[v] == v_meta
                    ):
                        total_flow += l
                k_meta = meta_commod[0]
                reconciliation_meta_out_flows[k_meta][u_meta] += total_flow
                reconciliation_meta_in_flows[k_meta][v_meta] += total_flow
                after_recon_flow[k_meta] = total_flow

            # computing reconciliation loss
            recon_loss = dict()
            for k_meta, _ in meta_commod_u_out_v_in:
                rmof_u = self.before_recon_meta_out_flow[k_meta][(u_meta, v_meta)]
                rmif_v = self.before_recon_meta_in_flow[k_meta][(u_meta, v_meta)]
                # these values may differ based on the R2 at each of the meta-nodes
                print(
                    "mk = {}, before_recon (u_out= {:.3f}, v_in= {:.3f}) after_recon= {:.3f}".format(
                        k_meta, rmof_u, rmif_v, after_recon_flow[k_meta]
                    )
                )
                recon_loss[k_meta] = min(rmof_u, rmif_v) - after_recon_flow[k_meta]

            if self.VERBOSE:
                self._print(
                    "Recon loss, Total={:.3f}; per meta-commod= {}".format(
                        sum(recon_loss.values()), recon_loss
                    )
                )

        # Kirchoff's
        if self.VERBOSE:
            self._print("\nKirchoff's")
        adjusted_meta_commodity_list = []
        self.r2_src_out_flows = self.divide_into_multi_commod_flows(
            self.r2_srcs_out_flow_lists, 0
        )
        self.r2_target_in_flows = self.divide_into_multi_commod_flows(
            self.r2_targets_in_flow_lists, 1
        )
        self.kirchoff_flow_per_commod = {}
        self._runtime_dict["kirchoffs"] = {}

        for (
            meta_commod_key,
            orig_commod_list_in_k_meta,
        ) in self.meta_commodity_dict.items():
            k_meta, (s_k_meta, t_k_meta, _) = meta_commod_key
            # self._print('\ns_k_meta: {}, t_k_meta: {}'.format(s_k_meta, t_k_meta))
            if len(self.r1_sol_dict[meta_commod_key]) == 0:
                # this seems unnecessary, but it fails an assert in _r3_lp
                adjusted_meta_commodity_list.append((k_meta, (s_k_meta, t_k_meta, 0.0)))
                for k, _ in orig_commod_list_in_k_meta:
                    self.kirchoff_flow_per_commod[k] = 0.0
                continue

            kirchoffs_solver = self._kirchoffs_lp(
                meta_commod_key, orig_commod_list_in_k_meta
            )
            kirchoffs_solver.solve_lp()
            if self.VERBOSE:
                self._print(
                    "\ns_k_meta: {}, t_k_meta: {}, obj val: {}".format(
                        s_k_meta, t_k_meta, kirchoffs_solver.obj_val
                    )
                )
            self._runtime_dict["kirchoffs"][
                (s_k_meta, t_k_meta)
            ] = kirchoffs_solver.model.Runtime
            flow_per_commod = self._extract_kirchoffs_sol(
                kirchoffs_solver.model, orig_commod_list_in_k_meta
            )

            adjusted_meta_demand = 0.0
            for k, _ in orig_commod_list_in_k_meta:
                adjusted_meta_demand += flow_per_commod[k]
                self.kirchoff_flow_per_commod[k] = flow_per_commod[k]

            adjusted_meta_commodity_list.append(
                (k_meta, (s_k_meta, t_k_meta, adjusted_meta_demand))
            )

        if self.VERBOSE:
            self._print("\nadjusted meta commodity list", adjusted_meta_commodity_list)

        # R3: Re-run R1, with a node capacity for each meta-commodity based on
        # the output of R2 and reconciliation
        if self.VERBOSE:
            self._print("\nR3")
        self.adjusted_meta_commodity_list = adjusted_meta_commodity_list
        r3_solver = self._r3_lp(adjusted_meta_commodity_list)
        r3_solver.gurobi_out = self.out.name.replace(".txt", "-r3.txt")
        r3_solver.solve_lp()
        self._runtime_dict["r3"] = r3_solver.model.Runtime
        self.r3_sol_dict = self.extract_sol_as_dict(
            r3_solver.model,
            adjusted_meta_commodity_list,
            self._r3_path_to_commod,
            self._r3_paths,
        )
        self._save_pkl(
            self.r3_sol_dict, self.out.name.replace(".txt", "-r3-sol-dict.pkl")
        )

        self.r3_obj_val = 0.0
        # Use original meta commod keys, instead of meta commod keys with
        # adjusted demands
        self.inter_sol_dict = {}
        for meta_commod_key, flow_list in self.r3_sol_dict.items():
            k_meta, (s_k_meta, _, _) = meta_commod_key
            flow_val = compute_in_or_out_flow(flow_list, 0, {s_k_meta})
            if self.DEBUG:
                self._print("reading r3: ", meta_commod_key, " flow_val: ", flow_val)
            self.r3_obj_val += flow_val
            self.inter_sol_dict[self.meta_commodity_list[k_meta]] = flow_list

        self._obj_val = sum(self.intra_obj_vals) + self.r3_obj_val

        if self.DEBUG:
            self.r3_sol_paths = self.extract_sol_by_paths(
                r3_solver.model, adjusted_meta_commodity_list, self._r3_path_to_commod
            )
            for x_k_meta in sorted(self.r3_sol_paths.keys()):
                self._print(
                    "xyz M{} = {}".format(x_k_meta, self.r3_sol_paths[x_k_meta].items())
                )

        self._print(
            "-->> Total flow= ",
            self._obj_val,
            " r3: ",
            self.r3_obj_val,
            " intra: ",
            sum(self.intra_obj_vals),
        )
        self._print("-->> Runtime= ", self.runtime_est(14))
        return self._obj_val

    ##############
    # PROPERTIES #
    ##############

    @property
    def partitioner(self):
        return self._partitioner

    @property
    def runtime(self):
        rts = self._runtime_dict
        total_time = (
            rts["r1"]
            + +sum(rts["r2"].values())
            + sum(rts["reconciliation"].values())
            + rts["r3"]
        )
        if "kirchoffs" in rts:
            total_time += sum(rts["kirchoffs"].values())
        return total_time

    @property
    def runtime_dict(self):
        return self._runtime_dict

    @property
    def obj_val(self):
        return self._obj_val

    @property
    def meta_inter_sol_dict(self):
        return self.inter_sol_dict

    @property
    def intra_sol_dict(self):
        return {
            commod_key: flow_list
            for sol_dict in self.intra_sols_dicts
            for commod_key, flow_list in sol_dict.items()
        }

    def compute_flow_per_inter_commod(self):
        # Calculate flow per commod that traverses between two or more meta-nodes
        flow_per_inter_commod = {}
        for (
            meta_commod_key,
            orig_commod_list_in_k_meta,
        ) in self.meta_commodity_dict.items():
            commod_ids = [key[0] for key in orig_commod_list_in_k_meta]
            meta_flow_list = self.inter_sol_dict[meta_commod_key]
            r3_meta_flow = compute_in_or_out_flow(
                meta_flow_list, 0, {meta_commod_key[-1][0]}
            )
            sum_of_kirchoff_flows = sum(
                self.kirchoff_flow_per_commod[k] for k in commod_ids
            )

            if abs(sum_of_kirchoff_flows - r3_meta_flow) < EPS:
                for k in commod_ids:
                    flow_per_inter_commod[k] = self.kirchoff_flow_per_commod[k]
            else:
                waterfall = waterfall_memoized()
                adjusted_commods = [
                    (
                        key[0],
                        (key[-1][0], key[-1][1], self.kirchoff_flow_per_commod[key[0]]),
                    )
                    for key in orig_commod_list_in_k_meta
                ]

                for k, _ in adjusted_commods:
                    waterfall_flow = waterfall(r3_meta_flow, k, adjusted_commods)
                    if waterfall_flow > EPS:
                        flow_per_inter_commod[k] = waterfall_flow
                    else:
                        flow_per_inter_commod[k] = 0.0

        return flow_per_inter_commod

    @property
    def sol_dict_as_paths(self):
        debug_sol_dict_as_paths = True
        if not debug_sol_dict_as_paths and hasattr(self, "_sol_dict_as_paths"):
            return self._sol_dict_as_paths

        self._sol_dict_as_paths = {}

        flow_per_inter_commod_after_r3 = self.compute_flow_per_inter_commod()

        meta_commod_to_meta_edge_fraction_of_r3_flow = defaultdict(
            lambda: defaultdict(float)
        )
        for (
            meta_commod_key,
            orig_commod_list_in_k_meta,
        ) in self.meta_commodity_dict.items():
            meta_flow_list = self.inter_sol_dict[meta_commod_key]
            r3_meta_flow = compute_in_or_out_flow(
                meta_flow_list, 0, {meta_commod_key[-1][0]}
            )

            for (u_meta, v_meta), meta_flow_val in meta_flow_list:
                meta_commod_to_meta_edge_fraction_of_r3_flow[meta_commod_key][
                    (u_meta, v_meta)
                ] += (meta_flow_val / r3_meta_flow)

        for meta_node_id, model in enumerate(self.r2_models):
            if model is None:
                continue
            multi_commodity_list = self.r2_mc_lists[meta_node_id]
            r2_all_paths = self.r2_paths[meta_node_id]

            commod_ids_to_meta_edge_to_path_ids = defaultdict(lambda: defaultdict(list))

            commod_ids_to_path_id_to_r2_flow = defaultdict(dict)

            for var in model.getVars():
                if not var.varName.startswith("fp") or var.x <= EPS:
                    continue
                match = re.match(r"fp(\d+)_mc(\d+)", var.varName)
                r2_path_id, mc_id = int(match.group(1)), int(match.group(2))
                srcs, targets, _, commod_ids = multi_commodity_list[mc_id]
                commod_ids = tuple(commod_ids)

                if (
                    srcs[0] not in self.virt_to_meta_dict
                    and targets[0] not in self.virt_to_meta_dict
                ):
                    commod_key = self.problem.commodity_list[commod_ids[0]]
                    if commod_key not in self._sol_dict_as_paths:
                        self._sol_dict_as_paths[commod_key] = {}
                        self._sol_dict_as_paths[commod_key][meta_node_id] = {}
                    path = tuple(r2_all_paths[r2_path_id])
                    # Store fraction of flow sent on this path
                    d_k = commod_key[-1][-1]
                    self._sol_dict_as_paths[commod_key][meta_node_id][path] = (
                        var.x / d_k
                    )
                    continue

                path = r2_all_paths[r2_path_id]
                start_node, end_node = path[0], path[-1]

                # Either srcs are virtual, targets are virtual, or both
                if targets[0] in self.virt_to_meta_dict:
                    # Leavers and transit
                    u_meta = meta_node_id
                    v_meta = (
                        self.virt_to_meta_dict[end_node]
                        if end_node in self.virt_to_meta_dict
                        else meta_node_id
                    )
                else:
                    # Enterers
                    u_meta = (
                        self.virt_to_meta_dict[start_node]
                        if start_node in self.virt_to_meta_dict
                        else meta_node_id
                    )
                    v_meta = meta_node_id

                commod_ids_to_meta_edge_to_path_ids[commod_ids][
                    (u_meta, v_meta)
                ].append(r2_path_id)
                commod_ids_to_path_id_to_r2_flow[commod_ids][r2_path_id] = var.x

            for (
                commod_ids,
                meta_edge_path_ids_dict,
            ) in commod_ids_to_meta_edge_to_path_ids.items():
                meta_commod_key = self.meta_commodity_list[
                    self.commod_id_to_meta_commod_id[commod_ids[0]]
                ]
                path_id_to_r2_flow = commod_ids_to_path_id_to_r2_flow[commod_ids]

                for (u_meta, v_meta), path_ids in meta_edge_path_ids_dict.items():
                    mc_r2_flow = sum(
                        path_id_to_r2_flow[path_id] for path_id in path_ids
                    )

                    for k in commod_ids:
                        commod_key = self.problem.commodity_list[k]
                        if commod_key not in self._sol_dict_as_paths:
                            self._sol_dict_as_paths[commod_key] = {}
                        if meta_node_id not in self._sol_dict_as_paths[commod_key]:
                            self._sol_dict_as_paths[commod_key][meta_node_id] = {}

                        # Every commodity needs to be present in the sol_dict, so we check
                        # if mc_r2_flow == 0.0 afterwards
                        if mc_r2_flow == 0.0:
                            continue
                        d_k = commod_key[-1][-1]

                        for path_id in path_ids:
                            # fractional flow per path per commod = (total flow per commod / demand per commod) * (flow per path in r2 / total flow in r2) * (meta flow on meta edge in r3 / total meta flow in r3)
                            fractional_flow_per_path_per_commod = (
                                (flow_per_inter_commod_after_r3[k] / d_k)
                                * (path_id_to_r2_flow[path_id] / mc_r2_flow)
                                * meta_commod_to_meta_edge_fraction_of_r3_flow[
                                    meta_commod_key
                                ][(u_meta, v_meta)]
                            )
                            if fractional_flow_per_path_per_commod == 0.0:
                                continue

                            path = tuple(r2_all_paths[path_id])
                            self._sol_dict_as_paths[commod_key][meta_node_id][
                                path
                            ] = fractional_flow_per_path_per_commod

        if debug_sol_dict_as_paths:
            print(
                "asserting demand constraints within each meta-node for all commods in sol_dict_as_paths"
            )
            for commod_key, meta_node_ids in self._sol_dict_as_paths.items():
                for meta_node_id in meta_node_ids:
                    assert (
                        sum(self._sol_dict_as_paths[commod_key][meta_node_id].values())
                        <= 1.0
                    )

        return self._sol_dict_as_paths

    @property
    def sol_dict(self):
        debug_sol_dict = False
        if not debug_sol_dict and hasattr(self, "_sol_dict"):
            return self._sol_dict

        self._sol_dict = self.intra_sol_dict

        flow_per_inter_commod_after_r3 = self.compute_flow_per_inter_commod()

        meta_commod_to_meta_edge_fraction_of_r3_flow = defaultdict(
            lambda: defaultdict(float)
        )
        for (
            meta_commod_key,
            orig_commod_list_in_k_meta,
        ) in self.meta_commodity_dict.items():
            meta_flow_list = self.inter_sol_dict[meta_commod_key]
            r3_meta_flow = compute_in_or_out_flow(
                meta_flow_list, 0, {meta_commod_key[-1][0]}
            )

            for (u_meta, v_meta), meta_flow_val in meta_flow_list:
                meta_commod_to_meta_edge_fraction_of_r3_flow[meta_commod_key][
                    (u_meta, v_meta)
                ] += (meta_flow_val / r3_meta_flow)

        for meta_node_id, model in enumerate(self.r2_models):
            if model is None:
                continue
            multi_commodity_list = self.r2_mc_lists[meta_node_id]
            r2_all_paths = self.r2_paths[meta_node_id]

            commod_ids_to_meta_edge_to_path_ids = defaultdict(lambda: defaultdict(list))

            commod_ids_to_path_id_to_r2_flow = defaultdict(dict)

            for var in model.getVars():
                if not var.varName.startswith("fp") or var.x <= EPS:
                    continue
                match = re.match(r"fp(\d+)_mc(\d+)", var.varName)
                r2_path_id, mc_id = int(match.group(1)), int(match.group(2))
                srcs, targets, _, commod_ids = multi_commodity_list[mc_id]
                commod_ids = tuple(commod_ids)

                if (
                    srcs[0] not in self.virt_to_meta_dict
                    and targets[0] not in self.virt_to_meta_dict
                ):
                    continue

                path = r2_all_paths[r2_path_id]
                start_node, end_node = path[0], path[-1]

                # Either srcs are virtual, targets are virtual, or both
                if targets[0] in self.virt_to_meta_dict:
                    # Leavers and transit
                    u_meta = meta_node_id
                    v_meta = (
                        self.virt_to_meta_dict[end_node]
                        if end_node in self.virt_to_meta_dict
                        else meta_node_id
                    )
                else:
                    # Enterers
                    u_meta = (
                        self.virt_to_meta_dict[start_node]
                        if start_node in self.virt_to_meta_dict
                        else meta_node_id
                    )
                    v_meta = meta_node_id

                commod_ids_to_meta_edge_to_path_ids[commod_ids][
                    (u_meta, v_meta)
                ].append(r2_path_id)
                commod_ids_to_path_id_to_r2_flow[commod_ids][r2_path_id] = var.x

            for (
                commod_ids,
                meta_edge_path_ids_dict,
            ) in commod_ids_to_meta_edge_to_path_ids.items():
                meta_commod_key = self.meta_commodity_list[
                    self.commod_id_to_meta_commod_id[commod_ids[0]]
                ]
                path_id_to_r2_flow = commod_ids_to_path_id_to_r2_flow[commod_ids]

                for (u_meta, v_meta), path_ids in meta_edge_path_ids_dict.items():
                    mc_r2_flow = sum(
                        path_id_to_r2_flow[path_id] for path_id in path_ids
                    )

                    for k in commod_ids:
                        commod_key = self.problem.commodity_list[k]
                        if commod_key not in self._sol_dict:
                            self._sol_dict[commod_key] = []

                        # Every commodity needs to be present in the sol_dict, so we check
                        # if mc_r2_flow == 0.0 afterwards
                        if mc_r2_flow == 0.0:
                            continue
                        for path_id in path_ids:
                            # flow per path per commod = total flow per commod * (flow per path in r2 / total flow in r2) * (meta flow on meta edge in r3 / total meta flow in r3)
                            flow_per_path_per_commod = (
                                flow_per_inter_commod_after_r3[k]
                                * (path_id_to_r2_flow[path_id] / mc_r2_flow)
                                * meta_commod_to_meta_edge_fraction_of_r3_flow[
                                    meta_commod_key
                                ][(u_meta, v_meta)]
                            )

                            if flow_per_path_per_commod == 0.0:
                                continue
                            for u, v in path_to_edge_list(r2_all_paths[path_id]):
                                if u in self.virt_to_meta_dict:
                                    continue
                                if v in self.virt_to_meta_dict:
                                    v_meta = self.virt_to_meta_dict[v]
                                    v = self.selected_inter_edges[
                                        (meta_node_id, v_meta)
                                    ][1]
                                self._sol_dict[commod_key].append(
                                    ((u, v), flow_per_path_per_commod)
                                )

        if debug_sol_dict:
            print("asserting flow conservation for all commods in sol_dict")
            for commod_key, flow_list in self._sol_dict.items():
                assert_flow_conservation(flow_list, commod_key)
        return self._sol_dict

    # Check:
    # 1) flow conservation constraints per commodity
    # 2) Check demand constraints are satisfied
    # 3) capacity constraints for every edge
    # 4) Our objective value is <= the objective value we found in solve()
    #
    # We also report the edge with the smallest remaining residual capacity
    def check_feasibility(self):
        self._print("Checking feasiblity of NCFlowSingleIter")
        obj_val = 0.0

        G_copy = self.problem.G.copy()
        self._print("checking flow conservation")
        for commod_key, flow_list in self.sol_dict.items():
            flow_for_commod = assert_flow_conservation(flow_list, commod_key)
            # assert demand constraints
            assert flow_for_commod <= commod_key[-1][-1] + EPS
            obj_val += flow_for_commod
            for (u, v), flow_val in flow_list:
                G_copy[u][v]["capacity"] -= flow_val
                # if G_copy[u][v]['capacity'] < 0.0:
                #     print(u, v, G_copy[u][v]['capacity'])
                assert G_copy[u][v]["capacity"] > -EPS

        assert (
            abs(obj_val - self.obj_val) <= EPS * self.obj_val or obj_val < self.obj_val
        )
        if obj_val < self.obj_val:
            print("delta in obj_val:", self.obj_val - obj_val)

        self._print("checking capacity constraints")
        edge_percent_cap_remaining = []
        for u, v, cap in G_copy.edges.data("capacity"):
            # if G_copy[u][v]['capacity'] < 0.0:
            #     print(u, v, G_copy[u][v]['capacity'])
            assert G_copy[u][v]["capacity"] > -EPS
            if self.problem.G[u][v]["capacity"] == 0.0:
                continue
            edge_percent_cap_remaining.append(
                (u, v, cap / self.problem.G[u][v]["capacity"])
            )

        bottleneck_edges = sorted(edge_percent_cap_remaining, key=lambda x: x[-1])
        print("Bottleneck edges")
        for min_u, min_v, min_cap in bottleneck_edges[:20]:
            min_u_meta_node = self._partition_vector[min_u]
            min_v_meta_node = self._partition_vector[min_v]
            if min_u_meta_node == min_v_meta_node:
                intra_or_inter = "intra, meta-node {}".format(min_u_meta_node)
            else:
                intra_or_inter = "inter, meta-edge ({}, {})".format(
                    min_u_meta_node, min_v_meta_node
                )
            print(
                "({}, {}), residual capacity: {}, {}".format(
                    min_u, min_v, min_cap, intra_or_inter
                )
            )
