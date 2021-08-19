#! /usr/bin/env python

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json
import sys
import os

OUTPUT_DIR = "../topologies"


def uni_rand(low=-1, high=1):
    return (high - low) * np.random.rand() + low


def read_graph_json(fname):
    assert fname.endswith(".json")
    with open(fname) as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)


def write_graph_json(fname, G):
    with open(fname, "w") as f:
        json.dump(json_graph.node_link_data(G), f)


def add_bi_edge(G, src, dest, capacity=None):
    G.add_edge(src, dest)
    G.add_edge(dest, src)
    if capacity:
        G[src][dest]["capacity"] = capacity
        G[dest][src]["capacity"] = capacity


################
# Toy Networks #
################
def two_srcs_from_meta_node():
    G = nx.DiGraph()
    G.add_node(0, label="0", pos=(-2, 2))
    G.add_node(1, label="1", pos=(-2, 1))
    G.add_node(2, label="2", pos=(0, 2))
    G.add_node(3, label="3", pos=(-1, 0))
    G.add_node(4, label="4", pos=(1, 0))

    add_bi_edge(G, 0, 2)
    add_bi_edge(G, 0, 1)
    add_bi_edge(G, 1, 3)
    add_bi_edge(G, 2, 3)
    add_bi_edge(G, 2, 4)
    add_bi_edge(G, 3, 4)

    return G


def dumbell_bottleneck_network():
    G = nx.DiGraph()
    G.add_node(0, label="0", pos=(-2, 2))
    G.add_node(1, label="1", pos=(-2, 1.5))

    G.add_node(2, label="2", pos=(0, 2))
    G.add_node(3, label="3", pos=(0, 1.5))

    G.add_node(4, label="4", pos=(-1, 1))
    G.add_node(5, label="5", pos=(-1, 0.5))

    G.add_node(6, label="6", pos=(1, 0))
    G.add_node(7, label="7", pos=(1, -0.5))

    # intra
    add_bi_edge(G, 0, 1)
    add_bi_edge(G, 2, 3)
    add_bi_edge(G, 4, 5)
    add_bi_edge(G, 6, 7)

    # inter
    add_bi_edge(G, 0, 2)
    add_bi_edge(G, 1, 4)
    add_bi_edge(G, 3, 4)
    add_bi_edge(G, 5, 6)
    add_bi_edge(G, 2, 6)

    return G


def toy_network_1():
    G = nx.DiGraph()
    G.add_node(0, label="0", pos=(-2, 2))
    G.add_node(1, label="1", pos=(-1, 0))
    G.add_node(2, label="2", pos=(-2, -2))
    G.add_node(3, label="3", pos=(2, 2))
    G.add_node(4, label="4", pos=(1, 0))
    G.add_node(5, label="5", pos=(2, -2))

    add_bi_edge(G, 0, 3)
    add_bi_edge(G, 0, 1)
    add_bi_edge(G, 1, 4)
    add_bi_edge(G, 1, 2)
    add_bi_edge(G, 2, 5)
    add_bi_edge(G, 3, 4)
    add_bi_edge(G, 4, 5)

    return G


def toy_network_2():
    G = nx.DiGraph()
    G.add_node(0, label="0", pos=(-2, 2))
    G.add_node(1, label="1", pos=(-1, 0))
    G.add_node(2, label="2", pos=(-2, -2))
    G.add_node(3, label="3", pos=(2, 2))
    G.add_node(4, label="4", pos=(1, 0))
    G.add_node(5, label="5", pos=(2, -2))

    G.add_node(6, label="6", pos=(-2, -4))
    G.add_node(7, label="7", pos=(-1, -5))
    G.add_node(8, label="8", pos=(-2, -6))
    G.add_node(9, label="9", pos=(2, -4))
    G.add_node(10, label="10", pos=(1, -5))
    G.add_node(11, label="11", pos=(2, -6))

    add_bi_edge(G, 0, 1)
    add_bi_edge(G, 0, 3)
    add_bi_edge(G, 1, 2)
    add_bi_edge(G, 1, 4)
    add_bi_edge(G, 2, 5)
    add_bi_edge(G, 3, 4)
    add_bi_edge(G, 4, 5)

    add_bi_edge(G, 6, 7)
    add_bi_edge(G, 6, 9)
    add_bi_edge(G, 7, 8)
    add_bi_edge(G, 7, 10)
    add_bi_edge(G, 8, 11)
    add_bi_edge(G, 9, 10)
    add_bi_edge(G, 10, 11)

    add_bi_edge(G, 2, 6)
    add_bi_edge(G, 1, 7)

    add_bi_edge(G, 4, 10)
    add_bi_edge(G, 5, 9)

    return G


def toy_network_3():
    G = nx.DiGraph()
    G.add_node(0, label="0", pos=(-2, 2))
    G.add_node(1, label="1", pos=(-1, 0))
    G.add_node(2, label="2", pos=(-2, -2))
    G.add_node(3, label="3", pos=(2, 2))
    G.add_node(4, label="4", pos=(1, 0))
    G.add_node(5, label="5", pos=(2, -2))

    add_bi_edge(G, 0, 1)
    add_bi_edge(G, 0, 3)
    add_bi_edge(G, 1, 2)
    add_bi_edge(G, 1, 4)
    add_bi_edge(G, 3, 4)
    add_bi_edge(G, 4, 5)
    return G


def bottleneck_network(cap=10.0, epsilon=1e-3):
    G = nx.DiGraph()
    G.add_node(0, label="0", pos=(-2, 2))
    G.add_node(1, label="1", pos=(-2, -2))
    G.add_node(2, label="2", pos=(2, 2))
    G.add_node(3, label="3", pos=(2, -2))

    add_bi_edge(G, 0, 1, capacity=epsilon)
    add_bi_edge(G, 0, 2, capacity=cap)
    add_bi_edge(G, 1, 3, capacity=cap)
    add_bi_edge(G, 2, 3, capacity=epsilon)
    return G


def erdos_renyi(num_nodes, prob, seed=None):
    G = nx.erdos_renyi_graph(num_nodes, prob, seed=seed, directed=False).to_directed()
    largest_scc = []
    for scc in nx.strongly_connected_component_subgraphs(G):
        if len(scc) > len(largest_scc):
            largest_scc = scc

    G = nx.convert_node_labels_to_integers(largest_scc.copy())
    for u, v in G.edges():
        G[u][v]["capacity"] = 1000.0  # Set every link to have 1000 Mbps capacity
    return G


#################
# Real Networks #
#################
def b4_teavar():
    G = nx.DiGraph()
    with open("../topologies/b4-teavar-topology.txt") as f:
        f.readline()  # skip header
        for line in f:
            vals = line.strip().split()
            # nodes are 1-indexed, capcity in Kbps
            from_node, to_node, cap = (
                int(vals[0]) - 1,
                int(vals[1]) - 1,
                float(vals[2]) / 1000.0,
            )
            G.add_edge(from_node, to_node, capacity=cap)
    return G


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "1":
        G = toy_network_1()
        fname = "toy-network.json"

    elif arg == "2":
        G = toy_network_2()
        fname = "toy-network-2.json"

    elif arg == "3":
        G = toy_network_2()
        fname = "toy-network-3.json"

    elif arg == "b4-teavar":
        G = b4_teavar()
        fname = "b4-teavar.json"

    elif arg == "bottleneck":
        G = bottleneck_network()
        fname = "bottleneck.json"

    elif arg == "two-srcs":
        G = two_srcs_from_meta_node()
        fname = "two-srcs.json"

    elif arg == "dumbell-bottleneck":
        G = dumbell_bottleneck_network()
        fname = "dumbell-bottleneck.json"

    elif arg == "erdos-renyi":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else np.random.randint(2 ** 31 - 1)
        G = erdos_renyi(1000, 0.005, seed=seed)
        fname = "erdos-renyi-{}.json".format(seed)

    data = json_graph.node_link_data(G)
    write_graph_json(os.path.join(OUTPUT_DIR, fname), G)
