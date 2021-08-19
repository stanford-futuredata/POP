import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .utils import uni_rand


def vis_graph(G, node_label="label", edge_label="capacity"):
    def get_node_attrs_or_default(G, attr, default_val):
        attr_dict = nx.get_node_attributes(G, attr)
        if not attr_dict:
            attr_dict = {}
        for node in G.nodes:
            if node not in attr_dict:
                if hasattr(default_val, "__call__"):
                    attr_dict[node] = default_val(node)
                else:
                    attr_dict[node] = default_val
        return attr_dict

    def random_pos(node):
        return (uni_rand(-3, 3), uni_rand(-3, 3))

    plt.figure(figsize=(14, 8))
    pos = get_node_attrs_or_default(G, "pos", random_pos)
    colors = get_node_attrs_or_default(G, "color", "yellow")
    colors = [colors[node] for node in G.nodes]
    nx.draw(G, pos, node_size=1000, node_color=colors)
    node_labels = get_node_attrs_or_default(G, node_label, str)
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    edge_labels = nx.get_edge_attributes(G, edge_label)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.8)
    plt.show()


def vis_partitions(G, partition_vector):
    # AT MOST 6 partitions
    COLORS = ["yellow", "green", "blue", "red", "orange", "purple"]
    assert np.max(partition_vector) <= len(COLORS)
    color_dict = {part_id: COLORS[part_id] for part_id in np.unique(partition_vector)}
    for node in G.nodes:
        G.nodes[node]["color"] = color_dict[partition_vector[node]]
    vis_graph(G)
