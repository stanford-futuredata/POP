import numpy as np
import networkx as nx
from collections import defaultdict
from ..graph_utils import compute_in_or_out_flow
import sys
from itertools import permutations
from networkx.algorithms.community import coverage as cov


# Serialize topology and partition_vector to graphml file, to visualize in
# Gephi. Each node's "id" field is assigned to its partition id; each edge's
# capacity is retained. (No other graph metadata is serialized.)
def serialize_to_graphml_for_gephi(prob, p_v):
    G = nx.DiGraph()
    for node, pos in prob.G.nodes.data("pos"):
        if pos is not None:
            G.add_node(node, id=p_v[node], latitude=pos[0], longitude=pos[1])
        else:
            G.add_node(node, id=p_v[node])

    for u, v, data in prob.G.edges.data():
        G.add_edge(u, v, capacity=data["capacity"])

    G = G.to_undirected()
    nx.write_graphml(
        G, prob.name.replace(".dot", ".graphml").replace(".json", ".graphml")
    )


def to_np_arr(arr):
    return arr if isinstance(arr, np.ndarray) else np.array(arr)


def coverage(prob, p_v):
    def convert_to_list_of_sets(p_v):
        return [
            set(np.argwhere(p_v == part_id).flatten()) for part_id in np.unique(p_v)
        ]

    return cov(prob.G, convert_to_list_of_sets(p_v))


def size_of_largest_partition(p_v):
    return max(np.bincount(p_v))


def is_partition_valid(prob, nodes_in_part):
    G_sub = prob.G.subgraph(nodes_in_part)
    tm = prob.traffic_matrix.tm
    for src, target in permutations(G_sub.nodes, 2):
        if tm[src, target] == 0.0:
            continue
        if not nx.has_path(G_sub, src, target):
            print(src, target)
            return False
    return True


def all_partitions_contiguous(prob, p_v):

    partition_vector = to_np_arr(p_v)
    for k in np.unique(partition_vector):
        if not is_partition_valid(
            prob, prob.G.subgraph(np.argwhere(partition_vector == k).flatten())
        ):
            print(k)
            return False
    return True


def count_meta_edges(G, p_v):
    partition_vector = to_np_arr(p_v)

    edge_cut_counts = defaultdict(int)
    edge_cut_capacities = defaultdict(float)
    for u, part_id in enumerate(partition_vector):
        for v in G.successors(u):
            if partition_vector[v] != part_id:
                print("({}, {})".format(u, v))
                edge_cut_counts[part_id] += 1
                edge_cut_capacities[part_id] += G[u][v]["capacity"]
    return dict(edge_cut_counts), dict(edge_cut_capacities)


def count_nodes_per_meta_node(partition_vector):
    return np.bincount(partition_vector)


def compute_total_intra_and_inter_flow(p_v, sol_dict):
    intra_flows = [0.0 for i in range(np.max(p_v) + 1)]
    inter_flows = defaultdict(float)
    for commod_key, flow_list in sol_dict.items():
        k, (s_k, t_k, _) = commod_key
        flow = compute_in_or_out_flow(flow_list, 0, {s_k})
        s_k_meta, t_k_meta = p_v[s_k], p_v[t_k]
        if s_k_meta == t_k_meta:
            intra_flows[s_k_meta] += flow
        else:
            inter_flows[(s_k_meta, t_k_meta)] += flow
    return intra_flows, dict(inter_flows)


# 1) local flow / avg # of meta-nodes
# 2) non-local flow / meta-edge capacities
# 3) avg # of meta-edges
# 4) max #  of meta-edges
# 5) Bayesian Information Criterion
def find_best_k(partition_constructor, problem, method="bic"):
    G = problem.G

    if method == "bic":
        k = 1
        best_k = k
        best_p_v = None
        best_bic = -999999999999999
        while k < int(len(G.nodes) / 2):
            print("Best K: ", best_k)
            print("Best BIC: ", best_bic)
            print()
            k += 1
            partitioner = partition_constructor(num_partitions=k)
            p_v = partitioner.partition(problem)
            if not all_partitions_contiguous(problem.G, p_v):
                print("{} not continguous".format(k))
                print()
                continue

            bic = partitioner.compute_bic()
            if bic > best_bic:
                best_k = k
                best_bic = bic
                best_p_v = p_v
            else:
                break
        return best_k, best_p_v, best_bic
    else:
        best_k = None
        best_p_v = None
        min_score = sys.maxsize

        # intra_flow, _ = compute_total_intra_and_inter_flow(
        #     np.zeros(len(G.nodes)), problem.traffic_matrix)
        # flow_per_node = intra_flow / len(G.nodes)
        # print('Flow per node:', flow_per_node)

        for k in range(2, int(len(G.nodes) / 2)):
            partitioner = partition_constructor(num_partitions=k)
            p_v = partitioner.partition(problem)
            if not all_partitions_contiguous(G, p_v):
                print("{} not contiguous".format(k))
                print()
                continue
            # nodes_per_meta_node = count_nodes_per_meta_node(p_v)
            # local_flow, non_local_flow = compute_local_and_non_local_flow(p_v, problem.traffic_matrix)

            # edge_cut_caps = list(edge_cut_caps_dict.values())
            # avg_nodes_per_meta_node = np.mean(nodes_per_meta_node)
            # median_meta_edges = np.median(edge_cut_counts)

            print("Best K: ", best_k)
            print("Min score: ", min_score)
            # print('Avg number of nodes per meta-node:', avg_nodes_per_meta_node)
            # print('local flow / avg nodes per meta-node:', local_flow / avg_nodes_per_meta_node)
            # print('non-local flow / total meta-edge capacity:', non_local_flow / np.sum(edge_cut_caps))
            # print('Avg number of meta-edges between meta-nodes:', mean_meta_edges)
            print()
            if method == "mean":
                edge_cut_counts_dict, edge_cut_caps_dict = count_meta_edges(G, p_v)
                edge_cut_counts = list(edge_cut_counts_dict.values())
                mean_meta_edges = np.mean(edge_cut_counts)
                if mean_meta_edges <= min_score:
                    min_score = mean_meta_edges
                    best_k = k
                    best_p_v = p_v
            elif method == "max":
                edge_cut_counts_dict, edge_cut_caps_dict = count_meta_edges(G, p_v)
                edge_cut_counts = list(edge_cut_counts_dict.values())
                max_meta_edges = np.max(edge_cut_counts)
                print("k: {}, max: {}".format(k, max_meta_edges))
                print()
                if max_meta_edges <= min_score:
                    min_score = max_meta_edges
                    best_k = k
                    best_p_v = p_v
            elif method == "intra":
                pass
            elif method == "inter":
                pass

        return best_k, best_p_v, min_score
