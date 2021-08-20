import numpy as np
import random
import math
from collections import defaultdict
import copy
from ...graph_utils import path_to_edge_list
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from .entity_splitting import split_entities
import time


def create_edges_onehot_dict(problem, pf_original, num_subproblems, split_fraction=0.1):
    paths_dict = pf_original.get_paths(problem)

    entity_list = [[k, u, v, d] for (k, (u, v, d)) in problem.commodity_list]
    split_entity_lists = split_entities(entity_list, split_fraction)

    # compose single list from all split entity lists
    split_entity_list = []
    for split_entity in split_entity_lists:
        split_entity_list += split_entity

    num_entities = len(split_entity_list)
    num_edges = len(problem.G.edges)
    enum_edges_dict = {}
    for i, edge in enumerate(problem.G.edges):
        enum_edges_dict[edge] = i

    # create dictionary of all edges used by each commodity
    com_path_edges_dict = defaultdict(list)
    min_demand = np.inf
    max_demand = 0

    for ind, [_, source, target, demand] in enumerate(split_entity_list):
        paths_array = paths_dict[(source, target)]

        if min_demand > demand:
            min_demand = demand
        if max_demand < demand:
            max_demand = demand

        # add an entry to the dict for each split, each of which has a fraction of demand
        for path in paths_array:
            com_path_edges_dict[(ind, source, target, demand)] += list(
                path_to_edge_list(path)
            )
    com_path_edges_onehot_dict = defaultdict(list)
    num_entities = len(com_path_edges_dict)
    np_data = np.zeros((num_entities, num_edges + 1))
    np_data_i = 0
    for (k, source, target, demand), edge_list in com_path_edges_dict.items():
        onehot_edge = [0] * num_edges
        for edge in edge_list:
            edge_i = enum_edges_dict[edge]
            onehot_edge[edge_i] = 1
            np_data[np_data_i, edge_i] = 1
        # add in normalized demand as a dimension
        norm_demand = (demand - min_demand) / (max_demand - min_demand)
        com_path_edges_onehot_dict[(k, source, target, demand)] = onehot_edge + [
            norm_demand
        ]
        np_data[np_data_i, -1] = norm_demand
        np_data_i += 1

    return com_path_edges_onehot_dict, np_data


# subproblem_list: list of lists, one for each subproblem, containing assigned entities
# input_set_dims: dictionary mapping entity to its dimensions
# calculate mean value of every dimension of each subproblem and return


def check_dims(subproblem_list, input_set_dims):

    num_subproblems = len(subproblem_list)
    num_dimensions = len(list(input_set_dims.values())[0])
    print(
        "checking split of "
        + str(num_dimensions)
        + " dimensions over "
        + str(num_subproblems)
        + " subproblems"
    )

    subproblem_dim_sums = [np.zeros(num_dimensions) for _ in range(num_subproblems)]

    for k in range(num_subproblems):
        subproblem_entities = subproblem_list[k]

        for entity in subproblem_entities:
            entity_dims = input_set_dims[entity]
            subproblem_dim_sums[k] = np.add(
                np.asarray(entity_dims), subproblem_dim_sums[k]
            )
        subproblem_dim_sums[k] = subproblem_dim_sums[k] / len(subproblem_entities)
        print("subproblem " + str(k) + ": " + str(subproblem_dim_sums[k]))
    return subproblem_dim_sums


def calc_cov_online(current_cov, current_means, num_entity, new_entity):
    num_dims = len(current_means)
    new_means = (current_means * num_entity + np.asarray(new_entity)) / (num_entity + 1)

    resid_row_array = np.asarray(
        [[d - new_means[i]] * num_dims for i, d in enumerate(new_entity)]
    )
    resid_col_array = np.transpose(resid_row_array)

    new_cov = current_cov * (num_entity - 1) + (num_entity / (num_entity - 1)) * (
        np.multiply(resid_row_array, resid_col_array)
    )
    new_cov = new_cov / num_entity

    return new_cov


# calculate the change in MSE between subproblem inputs and all inputs covariance
def calc_dist_cov_change(
    input_set_dims, new_entity, origin_dist_covs, num_entity_mean, current_covs
):
    num_dims = len(new_entity)
    # print("old covs: " + str(current_covs))
    num_entity_in_sp = num_entity_mean[0]
    new_covs = np.zeros((num_dims, num_dims))
    if num_entity_in_sp > 0:

        # calculate it from scratch for the first 50 elements
        if num_entity_in_sp < 2:
            input_set_dims_new = copy.deepcopy(input_set_dims)
            for d in range(num_dims):
                input_set_dims_new[d] += [new_entity[d]]
            new_covs = np.cov(input_set_dims_new)
        # use online update estimate
        else:
            new_covs = calc_cov_online(
                current_covs, num_entity_mean[1], num_entity_mean[0], new_entity
            )

    old_mse = ((current_covs - origin_dist_covs) ** 2).mean(axis=None)
    new_mse = ((new_covs - origin_dist_covs) ** 2).mean(axis=None)

    dist_diff = old_mse - new_mse
    return dist_diff, new_covs


# Compute the change in distance (2-norm of dimensional means)
# from the original problem inputs when adding new entity
# TODO: better distance metric that considers covariance?
def calc_dist_mean_change(input_set_dims, new_entity, origin_dist, num_entity_mean):
    num_dims = len(new_entity)
    num_entity = num_entity_mean[0]
    current_means = num_entity_mean[1]

    # new_means = (current_means*num_entity + np.asarray(new_entity))/(num_entity+1)
    new_means = current_means + np.asarray(new_entity)

    sq_sum_distance_new = np.sum(np.square((new_means - origin_dist) / origin_dist))
    sq_sum_distance = np.sum(np.square((current_means - origin_dist) / origin_dist))

    return math.sqrt(sq_sum_distance) - math.sqrt(sq_sum_distance_new)


def two_choice(input_dict, k, verbose=False, method="means"):

    num_inputs = len(input_dict)
    num_dimensions = len(list(input_dict.values())[0])

    # original_dist_dict: keys are dimension indices (0..d), values are frequency
    original_dist_means_array = np.zeros(num_dimensions)
    original_dist_inputs_by_dim = []
    for d in range(num_dimensions):
        inputs = [val[d] for val in input_dict.values()]
        original_dist_inputs_by_dim.append(inputs)
        sum_d = sum(inputs)
        # original dist will reflect the sum of each dimension, divided by the number of subproblems
        original_dist_means_array[d] = sum_d / k  # /(num_inputs*1.0)

    original_dist_cov = np.cov(original_dist_inputs_by_dim)

    # subproblem_dim_lists has k lists, one for each subproblem. Within each list are d lists,
    # each containing the value of a dimension for each entity assigned to that subproblem
    subproblem_dim_lists = [[[] for _ in range(num_dimensions)] for _ in range(k)]

    # subproblem_entity_assignments is a list of lists
    subproblem_entity_assignments = [[] for _ in range(k)]

    # Assign each entity to the sub-problem that would have their distance from the
    # original distribution shrink the most.
    num_assigned = 0
    subproblem_num_entity_means = [[0, np.zeros(num_dimensions)] for _ in range(k)]
    subproblem_covs = [np.zeros((num_dimensions, num_dimensions)) for _ in range(k)]
    sp_ids = [i for i in range(k)]
    for entity, dims in input_dict.items():
        if num_assigned % int(num_inputs / 4) == 0:
            print("Assigned " + str(num_assigned) + " entities")
        max_dist_change = -np.inf
        max_dist_sp = 0
        updated_cov = None

        # choose 2 random subproblems to compare, as long as they aren't equal or full
        num_sp_left = len(sp_ids)
        if num_sp_left == 1:
            max_dist_sp = sp_ids[0]
        else:
            random_sp1_id = random.randint(0, num_sp_left - 1)
            random_sp2_id = random.randint(0, num_sp_left - 1)
            while random_sp1_id == random_sp2_id:
                random_sp2_id = random.randint(0, num_sp_left - 1)
            random_sp1 = sp_ids[random_sp1_id]
            random_sp2 = sp_ids[random_sp2_id]

            for sp_index in [random_sp1, random_sp2]:

                # skip those that have more than equal share of currently assigned entities
                # if len(subproblem_entity_assignments[sp_index]) > (num_inputs+1.02)/(k*1.0):
                #    continue
                dist_change = 0
                if method == "means":
                    dist_change = calc_dist_mean_change(
                        subproblem_dim_lists[sp_index],
                        dims,
                        original_dist_means_array,
                        subproblem_num_entity_means[sp_index],
                    )
                elif method == "covs":
                    dist_change, new_cov = calc_dist_cov_change(
                        subproblem_dim_lists[sp_index],
                        dims,
                        original_dist_cov,
                        subproblem_num_entity_means[sp_index],
                        subproblem_covs[sp_index],
                    )
                # print("subproblem " + str(i) + ", dist change: " + str(dist_change))
                if dist_change >= max_dist_change:
                    max_dist_change = dist_change
                    max_dist_sp = sp_index
                    if method == "covs":
                        updated_cov = new_cov

            if method == "cov":
                subproblem_covs[max_dist_sp] = new_cov

        subproblem_entity_assignments[max_dist_sp].append(entity)
        if len(subproblem_entity_assignments[max_dist_sp]) > ((num_inputs) * 1.01) / (
            k * 1.0
        ):
            sp_ids.remove(max_dist_sp)
            # print("removed sp " + str(max_dist_sp))
        # update means to reflect entity assignment
        for d in range(num_dimensions):
            subproblem_dim_lists[max_dist_sp][d].append(dims[d])

        num_entity = subproblem_num_entity_means[max_dist_sp][0]
        dim_means = subproblem_num_entity_means[max_dist_sp][1]
        subproblem_num_entity_means[max_dist_sp][0] += 1
        # subproblem_num_entity_means[max_dist_sp][1] = (dim_means*num_entity + np.asarray(dims))/(num_entity+1)
        subproblem_num_entity_means[max_dist_sp][1] = dim_means + np.asarray(dims)

        num_assigned += 1

        if verbose:
            print(subproblem_dim_lists)
            print(subproblem_entity_assignments)
            print("\n")
    return subproblem_entity_assignments


# compute cluster using input data
def compute_precluster(data, num_clusters, categorical_indices=None):
    if categorical_indices is None:
        kp = KMeans(n_clusters=num_clusters)
        kp.fit(data)
    else:
        kp = KPrototypes(
            n_clusters=num_clusters,
            init="Cao",
            n_init=1,
            verbose=1,
            n_jobs=24,
            max_iter=5,
        )
        clusters = kp.fit(data, categorical=categorical_indices)
    return kp


# cluster data according to provided precluster (or compute it on the fly)
def cluster(data, k, precluster, categorical):
    # compute clusters, which is a list of cluster ids, one for each data item

    start_time = time.time()
    clusters = precluster.predict(data, categorical)
    print("--- %s seconds ---" % (time.time() - start_time))
    # print(clusters)

    # subproblem_entity_assignments is a list of lists
    subproblem_entity_assignments = [[] for _ in range(k)]

    num_clusters = precluster.n_clusters
    cluster_lists = [[] for _ in range(num_clusters)]
    for i, j in enumerate(clusters):
        cluster_lists[j].append(i)

    for cluster_list in cluster_lists:
        np.random.shuffle(cluster_list)
        # print(cluster_list)

    # equally assign items in each cluster across subproblems
    sp_roundrobin_id = 0
    for cluster in cluster_lists:
        for item in cluster:
            subproblem_entity_assignments[sp_roundrobin_id].append(item)
            sp_roundrobin_id = (sp_roundrobin_id + 1) % k

    # print(subproblem_entity_assignments)
    return subproblem_entity_assignments


# input_dict: keys are entities and values are (ordered) list of entity dimensions,
# k: number of subproblems
def split_generic(
    input_dict,
    k,
    verbose=False,
    method="means",
    precluster=None,
    np_data=None,
    categorical=None,
):

    if method == "cluster":
        subproblem_entity_assignments = cluster(
            np_data, k, precluster=precluster, categorical=categorical
        )
    elif method == "means" or method == "covs":
        subproblem_entity_assignments = two_choice(
            input_dict, k, verbose=verbose, method=method
        )

    return subproblem_entity_assignments
