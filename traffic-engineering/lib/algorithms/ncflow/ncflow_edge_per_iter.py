from ...config import TOPOLOGIES_DIR
from ..abstract_formulation import AbstractFormulation, Objective
from ...path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
from ...graph_utils import (
    compute_residual_problem,
    path_to_edge_list,
    assert_flow_conservation,
)
from .ncflow_single_iter import NCFlowSingleIter as NcfSi
from .counter import Counter
from ...partitioning.utils import all_partitions_contiguous

from itertools import product
from collections import defaultdict
from sys import maxsize

import networkx as nx
import numpy as np

import hashlib
import os
import pickle


PATHS_DIR = os.path.join(TOPOLOGIES_DIR, "paths", "ncflow-edge-per-iter")
R1_PATHS_DIR = PATHS_DIR + "/{}/{}/r1"
R2_PATHS_DIR = PATHS_DIR + "/{}/{}/r2"


class NCFlowEdgePerIter(AbstractFormulation):

    MAX_NUM_ITERS = 6

    @classmethod
    def new_total_flow(
        cls, num_paths, edge_disjoint=True, dist_metric="inv-cap", out=None, **args
    ):
        return cls(
            objective=Objective.TOTAL_FLOW,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=True,
            VERBOSE=False,
            out=out,
            **args
        )

    def __init__(
        self,
        *,
        objective,
        num_paths,
        edge_disjoint,
        dist_metric,
        DEBUG,
        VERBOSE,
        out=None,
        **args
    ):
        super().__init__(objective=objective, DEBUG=DEBUG, VERBOSE=VERBOSE, out=out)
        if dist_metric != "inv-cap" and dist_metric != "min-hop":
            raise Exception(
                'invalid distance metric: {}; only "inv-cap" and "min-hop" are valid choices'.format(
                    dist_metric
                )
            )
        self._num_paths = num_paths
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric
        self._args = args
        self.max_num_iters = self.MAX_NUM_ITERS

    def divide_problem_into_partitions(self, problem, partition_vector):
        G_meta_no_edges = nx.DiGraph()
        orig_G = problem.G
        commodity_list = problem.commodity_list

        for partition_id in np.unique(partition_vector):
            nodes = np.argwhere(partition_vector == partition_id).flatten()
            G_meta_no_edges.add_node(
                partition_id,
                label=str(partition_id),
                # position of the meta-node is the average position of all
                # the constituent nodes
                pos=np.mean(
                    [
                        orig_G.nodes.data("pos")[n]
                        if orig_G.nodes.data("pos")[n]
                        else (0.0, 0.0)
                        for n in nodes
                    ],
                    axis=0,
                ),
            )

        subgraph_dict = defaultdict(nx.DiGraph)
        intra_commods_dict = defaultdict(list)
        meta_edge_dict = defaultdict(list)
        meta_commodity_dict = defaultdict(list)

        for u, u_meta in enumerate(partition_vector):
            G_subgraph = subgraph_dict[u_meta]
            G_subgraph.add_node(u, **orig_G.nodes[u])
            for v in orig_G.successors(u):
                orig_cap = orig_G[u][v]["capacity"]
                v_meta = partition_vector[v]
                if u_meta == v_meta:  # intra-edge
                    G_subgraph.add_node(v, **orig_G.nodes[v])
                    G_subgraph.add_edge(u, v, capacity=orig_cap)
                else:  # inter-edge
                    meta_edge_dict[(u_meta, v_meta)].append((u, v))

        # mapping from meta-node id to (v_hat_in id , v_hat_out _id)
        meta_to_virt_dict = {}
        virt_to_meta_dict = {}  # mapping from v_hat id to meta-node id
        v_hat_i = max(orig_G.nodes) + 1
        for v_meta in G_meta_no_edges.nodes():
            meta_to_virt_dict[v_meta] = (v_hat_i, v_hat_i + 1)  # in and out
            virt_to_meta_dict[v_hat_i] = v_meta
            virt_to_meta_dict[v_hat_i + 1] = v_meta
            v_hat_i += 2
        if self.VERBOSE:

            def pretty_print_str(m_v_dict):
                to_print = [
                    "{}: ({} (in), {} (out))".format(k, v[0], v[1])
                    for k, v in m_v_dict.items()
                ]
                return "{" + ", ".join(to_print) + "}"

            self._print(
                "\nmeta -> virtual: {}".format(pretty_print_str(meta_to_virt_dict))
            )

        for v_meta in G_meta_no_edges.nodes:
            G_meta_no_edges.nodes[v_meta]["label"] += " ({}".format(
                meta_to_virt_dict[v_meta]
            )

        # For each s_k, t_k, d_k in the commodity list:
        #   If s_k and t_k belong to different partitions: // meta-commodity
        #       Add commodity to meta-commodity dict for meta(s_k), meta(t_k)
        #   Else: // intra flow
        #       Add commodity to subgraph commodity list for meta(s_k)
        for k, (s_k, t_k, d_k) in commodity_list:
            s_k_meta = partition_vector[s_k]
            t_k_meta = partition_vector[t_k]
            if s_k_meta != t_k_meta:
                meta_commodity_dict[(s_k_meta, t_k_meta)].append((k, (s_k, t_k, d_k)))
            else:
                intra_commods_dict[s_k_meta].append((k, (s_k, t_k, d_k)))

        meta_commodity_dict = {
            (k_meta, (u, v, sum([d_i for (_, (_, _, d_i)) in c_l]))): c_l
            for k_meta, ((u, v), c_l) in enumerate(meta_commodity_dict.items())
        }

        self.G_meta_no_edges = G_meta_no_edges
        self.meta_edge_dict = meta_edge_dict
        self.meta_commodity_dict = meta_commodity_dict
        self.commod_id_to_meta_commod_id = {
            commod_key[0]: meta_commod_key[0]
            for meta_commod_key, c_l in meta_commodity_dict.items()
            for commod_key in c_l
        }
        self.meta_commodity_list = list(self.meta_commodity_dict.keys())

        self.subgraphs = [
            G_subgraph for part_id, G_subgraph in sorted(subgraph_dict.items())
        ]
        self.intra_commods = intra_commods_dict

        self.meta_to_virt_dict = meta_to_virt_dict
        self.virt_to_meta_dict = virt_to_meta_dict

    def init_data_structures(self):
        self.G_metas = [self.G_meta_no_edges.copy() for _ in range(self.max_num_iters)]
        self.r2_G_hats = [
            [G_subgraph.copy() for G_subgraph in self.subgraphs]
            for _ in range(self.max_num_iters)
        ]
        # Note: the next two probably don't need to be a set per iteration, since they won't change. But
        # let's do it anyways to make the code less confusing
        self.all_u_hat_ins = [
            [set() for _ in self.G_metas[0].nodes()] for _ in range(self.max_num_iters)
        ]
        self.all_v_hat_outs = [
            [set() for _ in self.G_metas[0].nodes()] for _ in range(self.max_num_iters)
        ]
        orig_G = self.problem.G

        for iter in range(self.max_num_iters):
            selected_inter_edges = self.selected_inter_edges[iter]
            for (u_meta, v_meta), (u, v) in selected_inter_edges.items():
                cap = orig_G[u][v]["capacity"]
                self.G_metas[iter].add_edge(u_meta, v_meta, capacity=cap)

                G_hat_u_meta = self.r2_G_hats[iter][u_meta]
                _, v_hat_out = self.meta_to_virt_dict[v_meta]
                self.all_v_hat_outs[iter][u_meta].add(v_hat_out)
                G_hat_u_meta.add_node(v_hat_out)
                G_hat_u_meta.add_edge(u, v_hat_out, capacity=cap)

                G_hat_v_meta = self.r2_G_hats[iter][v_meta]
                u_hat_in, _ = self.meta_to_virt_dict[u_meta]
                self.all_u_hat_ins[iter][v_meta].add(u_hat_in)
                G_hat_v_meta.add_node(u_hat_in)
                G_hat_v_meta.add_edge(u_hat_in, v, capacity=cap)

    def update_data_structures_for_residual_problem(self, iter, residual_problem):
        # First update G_meta and selected_inter_edges
        for u_meta, v_meta in self.G_metas[iter].edges:
            u, v = self.selected_inter_edges[iter][(u_meta, v_meta)]
            new_cap = residual_problem.G[u][v]["capacity"]
            self.G_metas[iter][u_meta][v_meta]["capacity"] = new_cap

        # Next, update each individual r2_G_hat
        for meta_node_id, r2_G_hat in enumerate(self.r2_G_hats[iter]):
            for u, v in r2_G_hat.edges():
                orig_u, orig_v = u, v
                if u in self.virt_to_meta_dict:
                    orig_u = self.selected_inter_edges[iter][
                        (self.virt_to_meta_dict[u], meta_node_id)
                    ][0]
                if v in self.virt_to_meta_dict:
                    orig_v = self.selected_inter_edges[iter][
                        (meta_node_id, self.virt_to_meta_dict[v])
                    ][1]
                r2_G_hat[u][v]["capacity"] = residual_problem.G[orig_u][orig_v][
                    "capacity"
                ]

        # Finally, update:
        #   meta_commodity_dict
        #   meta_commodity_list
        #   intra_commods
        #   commod_id_to_meta_commod_id
        intra_commods_dict = defaultdict(list)
        meta_commodity_dict = defaultdict(list)
        for k, (s_k, t_k, d_k) in residual_problem.commodity_list:
            s_k_meta = self._partition_vector[s_k]
            t_k_meta = self._partition_vector[t_k]
            if s_k_meta != t_k_meta:
                meta_commodity_dict[(s_k_meta, t_k_meta)].append((k, (s_k, t_k, d_k)))
            else:
                intra_commods_dict[s_k_meta].append((k, (s_k, t_k, d_k)))

        meta_commodity_dict = {
            (k_meta, (u, v, sum([d_i for (_, (_, _, d_i)) in c_l]))): c_l
            for k_meta, ((u, v), c_l) in enumerate(meta_commodity_dict.items())
        }
        self.commod_id_to_meta_commod_id = {
            commod_key[0]: meta_commod_key[0]
            for meta_commod_key, c_l in meta_commodity_dict.items()
            for commod_key in c_l
        }

        self.meta_commodity_dict = meta_commodity_dict
        self.meta_commodity_list = list(self.meta_commodity_dict.keys())
        self.intra_commods = intra_commods_dict

    def select_inter_edges(self):
        selected_inter_edges = [{} for _ in range(self.max_num_iters)]
        orig_G = self.problem.G

        for (u_meta, v_meta), inter_edge_list_no_cap in self.meta_edge_dict.items():
            self._print("inter_edge_list:", inter_edge_list_no_cap)
            inter_edge_list = [
                (u, v, orig_G[u][v]["capacity"]) for u, v in inter_edge_list_no_cap
            ]
            # sort from highest capacity to lowest capacity
            highest_cap_inter_edges = sorted(
                inter_edge_list, key=lambda x: x[-1], reverse=True
            )[: self.max_num_iters]
            # if there's less than `max_num_iters` inter-edges, we'll recycle the edges until we reach `max_num_iters`
            while len(highest_cap_inter_edges) < self.max_num_iters:
                highest_cap_inter_edges += highest_cap_inter_edges
            highest_cap_inter_edges = highest_cap_inter_edges[: self.max_num_iters]

            for iter, (u, v, _) in enumerate(highest_cap_inter_edges):
                selected_inter_edges[iter][(u_meta, v_meta)] = (u, v)

        return selected_inter_edges

    def hash_partition(self, iter):
        partition_vector_str = "-".join(str(x) for x in self._partition_vector)
        sorted_inter_edges = sorted(
            [
                (u_meta, v_meta, u, v)
                for (u_meta, v_meta), (u, v) in self.selected_inter_edges[iter].items()
            ]
        )
        inter_edges_str = "--".join(
            "-".join(str(x) for x in group) for group in sorted_inter_edges
        )
        # raw fname is too long for Unix filesystem, so we compute md5 digest
        return hashlib.md5(
            (partition_vector_str + inter_edges_str).encode("utf-8")
        ).hexdigest()

    ############
    # R1 PATHS #
    ############
    @staticmethod
    def r1_paths_full_fname(
        problem_name, hash_partition_str, num_paths, edge_disjoint, dist_metric
    ):
        paths_dir = R1_PATHS_DIR.format(problem_name, hash_partition_str)
        if not os.path.exists(paths_dir):
            os.makedirs(paths_dir)
        return os.path.join(
            paths_dir,
            "r1_{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl".format(
                num_paths, edge_disjoint, dist_metric
            ),
        )

    # Use selected inter edges for zeroth iteration to compute R1 paths, even though edge capacities will change
    # afterwards
    def compute_r1_paths(self):
        full_fname = NCFlowEdgePerIter.r1_paths_full_fname(
            self.problem.name,
            self.hash_partition(0),
            self._num_paths,
            self.edge_disjoint,
            self.dist_metric,
        )

        G_meta = graph_copy_with_edge_weights(self.G_metas[0], self.dist_metric)
        paths_dict = {}

        from_nodes = set([node for node, degree in G_meta.out_degree() if degree > 0])
        to_nodes = set([node for node, degree in G_meta.in_degree() if degree > 0])
        for s_k_meta, t_k_meta in product(from_nodes, to_nodes):
            if s_k_meta != t_k_meta:
                paths = find_paths(
                    G_meta, s_k_meta, t_k_meta, self._num_paths, self.edge_disjoint
                )
                paths_dict[(s_k_meta, t_k_meta)] = paths

        self._print("saving R1 paths to pickle file: ", full_fname)
        with open(full_fname, "wb") as w:
            pickle.dump(paths_dict, w)
        return paths_dict

    def get_all_r1_paths(self):
        full_fname = NCFlowEdgePerIter.r1_paths_full_fname(
            self.problem.name,
            self.hash_partition(0),
            self._num_paths,
            self.edge_disjoint,
            self.dist_metric,
        )

        self._print("Loading R1 paths from pickle file", full_fname)
        try:
            with open(full_fname, "rb") as f:
                paths_dict = pickle.load(f)
                for key, paths in paths_dict.items():
                    paths_no_cycles = [remove_cycles(path) for path in paths]
                    paths_dict[key] = paths_no_cycles
                self._print("paths_dict size:", len(paths_dict))
        except FileNotFoundError:
            return self.compute_r1_paths()

        from_nodes = set(
            [node for node, degree in self.G_metas[0].out_degree() if degree > 0]
        )
        to_nodes = set(
            [node for node, degree in self.G_metas[0].in_degree() if degree > 0]
        )
        for s_k_meta, t_k_meta in product(from_nodes, to_nodes):
            if s_k_meta != t_k_meta and (s_k_meta, t_k_meta) not in paths_dict:
                raise Exception(
                    "No path from {} to {} in r1 paths dict!".format(s_k_meta, t_k_meta)
                )
        self._print()
        return paths_dict

    ############
    # R2 PATHS #
    ############
    @staticmethod
    def r2_paths_full_fname(
        meta_node_id,
        problem_name,
        hash_partition_str,
        num_paths,
        edge_disjoint,
        dist_metric,
    ):
        paths_dir = R2_PATHS_DIR.format(problem_name, hash_partition_str)
        if not os.path.exists(paths_dir):
            os.makedirs(paths_dir)
        return os.path.join(
            paths_dir,
            "meta-node-{}_{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl".format(
                meta_node_id, num_paths, edge_disjoint, dist_metric
            ),
        )

    def compute_r2_paths_for_meta_node(self, meta_node_id, iter):
        full_fname = NCFlowEdgePerIter.r2_paths_full_fname(
            meta_node_id,
            self.problem.name,
            self.hash_partition(iter),
            self._num_paths,
            self.edge_disjoint,
            self.dist_metric,
        )

        G_hat = self.r2_G_hats[iter][meta_node_id]
        all_v_hat_in = self.all_u_hat_ins[iter][meta_node_id]
        all_v_hat_out = self.all_v_hat_outs[iter][meta_node_id]
        subgraph_nodes = np.argwhere(self._partition_vector == meta_node_id).flatten()
        G_hat_weighted = graph_copy_with_edge_weights(G_hat, self.dist_metric)
        from_nodes = set(subgraph_nodes).union(all_v_hat_in)
        to_nodes = set(subgraph_nodes).union(all_v_hat_out)

        paths_dict = {}
        for s, t in product(from_nodes, to_nodes):
            if s != t:
                try:
                    paths_dict[(s, t)] = find_paths(
                        G_hat_weighted,
                        s,
                        t,
                        self._num_paths,
                        disjoint=self.edge_disjoint,
                    )
                except Exception as e:
                    self._print(e)
                    self._print("can't find paths: ", s, " -> ", t)
        self._print("Saving R2 paths to pickle file:", full_fname)
        with open(full_fname, "wb") as w:
            pickle.dump(paths_dict, w)
        return paths_dict

    def get_all_r2_paths_for_meta_node(self, meta_node_id, iter):
        full_fname = NCFlowEdgePerIter.r2_paths_full_fname(
            meta_node_id,
            self.problem.name,
            self.hash_partition(iter),
            self._num_paths,
            self.edge_disjoint,
            self.dist_metric,
        )

        self._print(
            "Loading R2 paths from pickle file, iter {}".format(iter), full_fname
        )
        try:
            with open(full_fname, "rb") as f:
                paths_dict = pickle.load(f)
                for key, paths in paths_dict.items():
                    paths_no_cycles = [remove_cycles(path) for path in paths]
                    paths_dict[key] = paths_no_cycles
                self._print("paths_dict size:", len(paths_dict))
        except FileNotFoundError:
            return self.compute_r2_paths_for_meta_node(meta_node_id, iter)

        all_v_hat_in = self.all_u_hat_ins[iter][meta_node_id]
        all_v_hat_out = self.all_v_hat_outs[iter][meta_node_id]
        subgraph_nodes = np.argwhere(self._partition_vector == meta_node_id).flatten()
        from_nodes = set(subgraph_nodes).union(all_v_hat_in)
        to_nodes = set(subgraph_nodes).union(all_v_hat_out)

        for s, t in product(from_nodes, to_nodes):
            if s != t:
                if (s, t) not in paths_dict:
                    raise Exception(
                        "No path from {} to {} in r2 paths dict for meta-node {}!".format(
                            s, t, meta_node_id
                        )
                    )

        self._print()
        return paths_dict

    # For each meta-commod for each iteration, we select the path with the lowest possible weight for that
    # meta-source and meta-target. Once all paths are existed, we repeat the process.
    def select_r1_paths(self):
        def compute_weight(G, path, dist_metric):
            if dist_metric == "inv-cap":
                caps = [G[u][v]["capacity"] for u, v in path_to_edge_list(path)]
                if 0.0 in caps:
                    return maxsize
                else:
                    return sum(1.0 / cap for cap in caps)
            elif dist_metric == "min-hop":
                return len(path_to_edge_list(path))
            else:
                raise Exception("invalid dist_metric: {}".format(dist_metric))

        selected_paths = [{} for _ in range(self.max_num_iters)]
        r1_paths_selected_so_far = {
            k_meta: set() for k_meta, _ in self.meta_commodity_list
        }
        for iter in range(self.max_num_iters):
            for k_meta, (s_k_meta, t_k_meta, _) in self.meta_commodity_list:
                all_paths_and_weights = sorted(
                    [
                        (path, compute_weight(self.G_metas[0], path, self.dist_metric))
                        for path in self.r1_paths_full_dict[(s_k_meta, t_k_meta)]
                    ],
                    key=lambda x: x[-1],
                )
                paths_to_choose_from = [
                    path
                    for path, _ in all_paths_and_weights
                    if tuple(path) not in r1_paths_selected_so_far[k_meta]
                ]
                if len(paths_to_choose_from) == 0:
                    paths_to_choose_from = [path for path, _ in all_paths_and_weights]
                    r1_paths_selected_so_far[k_meta].clear()

                selected_path = paths_to_choose_from[0]
                selected_paths[iter][(s_k_meta, t_k_meta)] = [selected_path]
                r1_paths_selected_so_far[k_meta].add(tuple(selected_path))

        return selected_paths

    def pre_solve(self, problem, partitioner):
        self._problem = problem
        if hasattr(self, "_sol_dict"):
            # invalidate previous sol dict
            del self._sol_dict

        # PARTITIONING #
        self._print("Generate partitioning")
        self._partition_vector = partitioner.partition(problem)
        if self.DEBUG:
            assert all_partitions_contiguous(problem, self._partition_vector)
        if len(np.unique(self._partition_vector)) != partitioner.num_partitions:
            if self.VERBOSE:
                self._print(
                    "{} partitions requested, but {} partitions generated".format(
                        partitioner.num_partitions,
                        len(np.unique(self._partition_vector)),
                    )
                )
                for i, part_id in enumerate(np.unique(self._partition_vector)):
                    if i == part_id:
                        continue
                    self._partition_vector[
                        np.argwhere(self._partition_vector == part_id).flatten()
                    ] = i
        self._print(self._partition_vector)
        self._save_txt(
            self._partition_vector,
            self.out.name.replace(".txt", "-partition-vector.txt"),
        )

        if self.VERBOSE:
            self._print("Partition network")
        self.divide_problem_into_partitions(problem, self._partition_vector)

        # PATH COMPUTATION #

        # First select the sequence of inter edges between each pair of meta-nodes for
        # *all* meta-commods for *all* iterations. The inter edge will change for each
        # iteration.
        self.selected_inter_edges = self.select_inter_edges()
        self.init_data_structures()

        # Then, compute the R1 paths that will be used for *all* iterations.
        # (We use G_meta from the very first iteration, even though the edge
        #  capacities will change.) Choose which R1 path we will use for
        # each iteration.
        self.r1_paths_full_dict = self.get_all_r1_paths()
        self.r1_path_assignments = self.select_r1_paths()

        # Last, we compute the R2 paths for each meta-node for *all* iterations
        self.r2_paths_dicts = [[] for _ in range(self.max_num_iters)]
        for meta_node_id in np.unique(self._partition_vector):
            for iter in range(self.max_num_iters):
                self.r2_paths_dicts[iter].append(
                    self.get_all_r2_paths_for_meta_node(meta_node_id, iter)
                )

    def solve(self, problem, partitioner):
        def get_ncflow_obj(iter):
            if self.out.name == "stdout" or self.out.name == "<stdout>":
                return NcfSi(
                    objective=self._objective,
                    DEBUG=self.DEBUG,
                    VERBOSE=self.VERBOSE,
                    out=self.out,
                    **self._args
                )
            else:
                log = open(
                    self.out.name.replace(".txt", "_iter_{}.txt".format(iter)), "w"
                )
                return NcfSi(
                    objective=self._objective,
                    DEBUG=self.DEBUG,
                    VERBOSE=self.VERBOSE,
                    out=log,
                    **self._args
                )

        self.pre_solve(problem, partitioner)

        # FLOW SOLVING #
        curr_prob = problem
        # orig_name = problem.name
        init_total_demand = curr_prob.total_demand
        self._ncflows = []
        self._problems = [problem]

        for iter in range(self.max_num_iters):
            print("iteration {}\n".format(iter))

            if iter > 0:
                self.update_data_structures_for_residual_problem(iter, curr_prob)

            # Retrieve which R1 path we will use for each meta-commodity for this iteration
            r1_paths_dict_current_iter = self.r1_path_assignments[iter]
            # Then run NCFlowSingleIter with those paths and the R2 paths we already computed
            nc = get_ncflow_obj(iter)
            nc.solve(
                curr_prob,
                self._partition_vector,
                self.G_metas[iter],
                r1_paths_dict_current_iter,
                self.r2_paths_dicts[iter],
                self.r2_G_hats[iter],
                self.all_u_hat_ins[iter],
                self.all_v_hat_outs[iter],
                self.intra_commods,
                self.meta_commodity_list,
                self.meta_commodity_dict,
                self.commod_id_to_meta_commod_id,
                self.virt_to_meta_dict,
                self.meta_to_virt_dict,
                self.selected_inter_edges[iter],
            )
            self._ncflows.append(nc)
            # Compute residual problem and iterate
            self._print("Computing residual problem after iteration {}".format(iter))
            curr_prob = compute_residual_problem(curr_prob.copy(), nc.sol_dict)
            self._problems.append(curr_prob)
            if nc.out.name != "stdout" and nc.out.name != "<stdout>":
                nc.out.close()

            # Stop early if we're within 5%
            if curr_prob.total_demand / init_total_demand < 0.05:
                self._print("within 5\% of optimality gap; stopping early")
                break

            flow_before_current_iter = sum(nc.obj_val for nc in self._ncflows[:-1])
            if (
                flow_before_current_iter > 0.0
                and nc.obj_val / flow_before_current_iter < 0.05
            ):
                self._print(
                    "less than 5\% improvement in last iteration; stopping early"
                )
                break

            if curr_prob.total_capacity == 0.0:
                self._print("total residual capacity equals 0.0")
                break

        self.num_iters = iter + 1

        if self._objective == Objective.TOTAL_FLOW:
            self._obj_val = sum(nc.obj_val for nc in self._ncflows)
        else:
            raise Exception(
                "no support for other Objectives besides Objective.TOTAL_FLOW"
            )

        return self._obj_val

    @property
    def sol_dict(self):
        if not hasattr(self, "_sol_dict"):
            # Initialize with commod -> empty list for every commod in the original
            # problem
            self._sol_dict = {commod: [] for commod in self.problem.commodity_list}
            # Since each NcfSi has its own commodity list, we use (s_k, t_k) as a
            # universal mapping
            src_target_pair_to_commod = {
                (commod[1][0], commod[1][1]): commod
                for commod in self.problem.commodity_list
            }
            for nc in self._ncflows:
                sol_dict = nc.sol_dict
                for (_, (s_k, t_k, _)), flow_list in sol_dict.items():
                    self._sol_dict[src_target_pair_to_commod[(s_k, t_k)]] += flow_list

        return self._sol_dict

    def check_feasibility(self):
        self._print("Checking feasibility of NCFlowEdgePerIter")
        obj_val = 0.0
        EPS = 1e-3

        G_copy = self.problem.G.copy()
        self._print("checking flow conservation")
        for nc in self._ncflows:
            for commod_key, flow_list in nc.sol_dict.items():
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
        self._print("Bottleneck edges")
        for min_u, min_v, min_cap in bottleneck_edges[:20]:
            min_u_meta_node = self._partition_vector[min_u]
            min_v_meta_node = self._partition_vector[min_v]
            if min_u_meta_node == min_v_meta_node:
                intra_or_inter = "intra, meta-node {}".format(min_u_meta_node)
            else:
                intra_or_inter = "inter, meta-edge ({}, {})".format(
                    min_u_meta_node, min_v_meta_node
                )
            self._print(
                "({}, {}), residual capacity: {}, {}".format(
                    min_u, min_v, min_cap, intra_or_inter
                )
            )

    @classmethod
    # Return total number of fib entries and max for any node in topology
    # NOTE: problem has to have a full TM matrix; otherwise, our path set
    # will have a problem
    def fib_entries(cls, problem, partitioner, num_paths, edge_disjoint, dist_metric):
        assert problem.is_traffic_matrix_full
        ncflow = cls.new_total_flow(
            num_paths=num_paths, edge_disjoint=edge_disjoint, dist_metric=dist_metric
        )
        ncflow.pre_solve(problem, partitioner)
        return ncflow.num_fib_entries_for_path_set()

    # Return (total # of fib entries, max # of fib entries)
    def num_fib_entries_for_path_set(self):
        self.fib_dict = defaultdict(dict)

        num_partitions = len(np.unique(self._partition_vector))

        r2_path_id_counters = [Counter() for _ in range(num_partitions)]
        r1_counter = Counter()

        for meta_node_id, intra_commods in self.intra_commods.items():
            r2_counter = r2_path_id_counters[meta_node_id]
            for k, (src, target, _) in intra_commods:
                commod_id_str = "k-{}".format(k)
                for iter in range(self.max_num_iters):
                    r2_paths = self.r2_paths_dicts[iter][meta_node_id][(src, target)]
                    r2_path_ids = [r2_counter[path] for path in r2_paths]
                    if commod_id_str not in self.fib_dict[src]:
                        self.fib_dict[src][commod_id_str] = set(r2_path_ids)
                    else:
                        for r2_path_id in r2_path_ids:
                            self.fib_dict[src][commod_id_str].add(r2_path_id)

                    for r2_path_id, r2_path in zip(r2_path_ids, r2_paths):
                        if r2_path_id in self.fib_dict[r2_path[0]]:
                            continue
                        for u, v in path_to_edge_list(r2_path):
                            assert r2_path_id not in self.fib_dict[u]
                            self.fib_dict[u][r2_path_id] = v

        for (
            (k_meta, (s_k_meta, t_k_meta, _)),
            orig_commod_list_in_k_meta,
        ) in self.meta_commodity_dict.items():
            meta_commod_id_str = "k-meta-{}".format(k)
            subgraph_src_nodes = np.argwhere(
                self._partition_vector == s_k_meta
            ).flatten()
            subgraph_target_nodes = np.argwhere(
                self._partition_vector == t_k_meta
            ).flatten()

            # Initialize all src nodes in src meta-node
            for src in subgraph_src_nodes:
                self.fib_dict[src][meta_commod_id_str] = {}

            # For each r1 path
            for iter in range(self.max_num_iters):
                r1_path = self.r1_path_assignments[iter][(s_k_meta, t_k_meta)][0]
                assert r1_path[0] == s_k_meta
                r1_path_id = r1_counter[r1_path]

                # Handle src meta-node
                # For this path, all srcs within this metanode will have this target
                virtual_target = self.meta_to_virt_dict[r1_path[1]][-1]
                r2_paths_dict = self.r2_paths_dicts[iter][s_k_meta]

                # For every src node in this src meta-node, look up the r2
                # paths between it and the virtual out node that we're targeting
                # for this path
                for src in subgraph_src_nodes:
                    r2_paths = r2_paths_dict[(src, virtual_target)]
                    r2_path_ids = [r2_counter[path] for path in r2_paths]
                    if r1_path_id not in self.fib_dict[src][meta_commod_id_str]:
                        self.fib_dict[src][meta_commod_id_str][r1_path_id] = set(
                            r2_path_ids
                        )
                    else:
                        for r2_path_id in r2_path_ids:
                            self.fib_dict[src][meta_commod_id_str][r1_path_id].add(
                                r2_path_id
                            )

                    for r2_path_id, r2_path in zip(r2_path_ids, r2_paths):
                        if (r1_path_id, r2_path_id) in self.fib_dict[r2_path[0]]:
                            continue
                        for u, v in path_to_edge_list(r2_path[:-1]):
                            assert (r1_path_id, r2_path_id) not in self.fib_dict[u]
                            self.fib_dict[u][(r1_path_id, r2_path_id)] = v

                        # For the very last edge that leaves the meta-node, look up the
                        # actual node that represents the virtual out node, and put that
                        # in the next hop entry
                        u = v
                        v = self.selected_inter_edges[iter][(s_k_meta, r1_path[1])][1]
                        self.fib_dict[u][(r1_path_id, r2_path_id, iter)] = v

                # Handle transit
                virtual_src = self.meta_to_virt_dict[s_k_meta][0]  # in virtual node
                for u_meta, v_meta in path_to_edge_list(r1_path[1:]):
                    virtual_target = self.meta_to_virt_dict[v_meta][-1]

                    #  Look up all the r2 paths between the virtual in node and the virtual out node
                    r2_paths_dict = self.r2_paths_dicts[iter][u_meta]
                    r2_paths = r2_paths_dict[(virtual_src, virtual_target)]
                    r2_path_ids = [r2_counter[path] for path in r2_paths]

                    for r2_path_id, r2_path in zip(r2_path_ids, r2_paths):
                        # We don't set the src entry in the virtual in node; instead, we place it
                        # at the very next node in the path, the ingress transit node
                        ingress_transit_node = r2_path[1]
                        if (
                            meta_commod_id_str
                            not in self.fib_dict[ingress_transit_node]
                        ):
                            self.fib_dict[ingress_transit_node][meta_commod_id_str] = {}
                        if (
                            r1_path_id
                            not in self.fib_dict[ingress_transit_node][
                                meta_commod_id_str
                            ]
                        ):
                            self.fib_dict[ingress_transit_node][meta_commod_id_str][
                                r1_path_id
                            ] = set()

                        self.fib_dict[ingress_transit_node][meta_commod_id_str][
                            r1_path_id
                        ].add(r2_path_id)

                        for u, v in path_to_edge_list(r2_path[1:-1]):
                            self.fib_dict[u][(r1_path_id, r2_path_id)] = v

                        # For the very last edge that leaves the meta-node, look up the
                        # actual node that represents the virtual out node, and put that
                        # in the next hop entry
                        u = v
                        v = self.selected_inter_edges[iter][(s_k_meta, r1_path[1])][1]
                        self.fib_dict[u][(r1_path_id, r2_path_id, iter)] = v

                    virtual_src = self.meta_to_virt_dict[u_meta][
                        0
                    ]  # advance virtual in node for next iteration

                # Handle target meta-node
                # For every target node in this target meta-node, look up the r2
                # paths between the current virtual in node and and it
                r2_paths_dict = self.r2_paths_dicts[iter][t_k_meta]
                for target in subgraph_target_nodes:
                    r2_paths = r2_paths_dict[(virtual_src, target)]
                    r2_path_ids = [r2_counter[path] for path in r2_paths]

                    for r2_path_id, r2_path in zip(r2_path_ids, r2_paths):
                        # We don't set the src entry in the virtual in node; instead, we place it
                        # at the very next node in the path, the ingress transit node
                        ingress_transit_node = r2_path[1]
                        if (
                            meta_commod_id_str
                            not in self.fib_dict[ingress_transit_node]
                        ):
                            self.fib_dict[ingress_transit_node][meta_commod_id_str] = {}
                        if (
                            r1_path_id
                            not in self.fib_dict[ingress_transit_node][
                                meta_commod_id_str
                            ]
                        ):
                            self.fib_dict[ingress_transit_node][meta_commod_id_str][
                                r1_path_id
                            ] = set()

                        self.fib_dict[ingress_transit_node][meta_commod_id_str][
                            r1_path_id
                        ].add(r2_path_id)

                        # Last node isn't a virtual node, so we encode the next hop till the end of the path
                        for u, v in path_to_edge_list(r2_path[1:]):
                            self.fib_dict[u][(r1_path_id, r2_path_id)] = v

        for node in self.fib_dict.keys():
            # Make sure never added any virtual nodes
            assert node in self.problem.G

        self.fib_dict = dict(self.fib_dict)
        fib_dict_counts = [len(self.fib_dict[k]) for k in self.fib_dict.keys()]
        return sum(fib_dict_counts), max(fib_dict_counts)

    @property
    def obj_val(self):
        return self._obj_val

    def runtime_est(self, num_threads):
        return sum(nc.runtime_est(num_threads) for nc in self._ncflows)
