import numpy as np
from collections import defaultdict

# sort commods from lowest demand to highest demand
# flow_remaining = flow_val
# demand_remaining = [d_k for commod]
# while len(demand_remaining) > 0:
#   peek at the first commod (the commod with the lowest demand)
#   flow_to_assign = min(flow_val / # commods, demand_remaining)
#   for every commod:
#       demand_remaining[commod] -= flow_to_assign
#       if demand_remaining[commod] == 0.0:
#           remove commod from demand_remaining
#           remove commod from sorted commods
#       flow_remaining -= flow_to_assign for every commod
#   if flow_remaining == 0.0:
#       break
# return demand - demand_remaining for k
def waterfall_memoized():
    # Memoize results in demand_satisfied
    demand_satisfied = {}

    def fn(flow_val, k, commods):
        if k in demand_satisfied:
            return demand_satisfied[k]

        EPS = 1e-6
        demand_remaining = {commod[0]: commod[-1][-1] for commod in commods}
        flow_remaining = flow_val
        sorted_commods = [
            commod[0] for commod in sorted(commods, key=lambda x: x[-1][-1])
        ]
        while len(demand_remaining) > 0:
            k_smallest = sorted_commods[0]
            flow_to_assign = min(
                flow_remaining / len(commods), demand_remaining[k_smallest]
            )
            for commod_id, (_, _, orig_demand) in commods:
                if commod_id not in demand_remaining:
                    continue
                demand_remaining[commod_id] -= flow_to_assign
                if abs(demand_remaining[commod_id] - 0.0) < EPS:
                    demand_satisfied[commod_id] = orig_demand
                    del demand_remaining[commod_id]
                    sorted_commods.remove(commod_id)
                flow_remaining -= flow_to_assign
            if abs(flow_remaining - 0.0) < EPS:
                break
        for commod_id, (_, _, orig_demand) in commods:
            if commod_id in demand_remaining:
                demand_satisfied[commod_id] = orig_demand - demand_remaining[commod_id]

        return demand_satisfied[k]

    return fn


# Convert nested defaultdict to dict
def nested_ddict_to_dict(ddict):
    for k, v in ddict.items():
        ddict[k] = dict(v)
    return dict(ddict)


# Converts {k1: [v1,... vn]} to {v1: k1,... vn: k1}
def reverse_dict_value_list(dict_of_list):
    return {v: k for k, vals in dict_of_list.items() for v in vals}


# Uniform random variable [low, high)
def uni_rand(low=-1, high=1):
    return (high - low) * np.random.rand() + low


def compute_max_link_util(G, sol_dict):
    total_flow_on_edge = defaultdict(float)
    for commod, flow_list in sol_dict.items():
        for (u, v), l in flow_list:
            total_flow_on_edge[(u, v)] += l
    max_edge = None
    max_util = 0.0
    for (u, v), total_flow in total_flow_on_edge.items():
        edge_util = total_flow / G[u][v]["capacity"]
        if edge_util > max_util:
            max_util = edge_util
            max_edge = (u, v)

    return max_edge, max_util


def link_util_stats(G, sol_dict):
    edge_flows = defaultdict(float)

    for flow_list in sol_dict.values():
        for (u, v), l in flow_list:
            edge_flows[(u, v)] += l

    edge_utils = {}
    for u, v, c_e in G.edges.data("capacity"):
        if c_e == 0.0:
            assert edge_flows[(u, v)] == 0.0
            edge_utils[(u, v)] = 0.0
        else:
            edge_utils[(u, v)] = edge_flows[(u, v)] / c_e

    values = list(edge_utils.values())

    return np.min(values), np.median(values), np.mean(values), np.max(values)
