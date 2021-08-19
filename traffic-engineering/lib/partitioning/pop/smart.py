from collections import defaultdict
from .abstract_pop_splitter import AbstractPOPSplitter
from ...graph_utils import path_to_edge_list
import numpy as np


class SmartSplitter(AbstractPOPSplitter):
    # paths_dict: key: (source, target), value: array of paths,
    #             where a path is a list of sequential nodes
    #             use lib.graph_utils.path_to_edge_list to get edges.
    def __init__(self, num_subproblems, paths_dict):
        super().__init__(num_subproblems)
        self._paths_dict = paths_dict

    def split(self, problem):
        com_list = problem.commodity_list

        max_demand = 100.0 / self._num_subproblems

        # create dictionary of all edges used by each commodity
        com_path_edges_dict = defaultdict(list)
        for k, (source, target, demand) in com_list:

            num_split_entity = 1
            if demand > max_demand:
                num_split_entity = min(
                    self._num_subproblems, int(np.ceil(demand / max_demand))
                )

            paths_array = self._paths_dict[(source, target)]
            for path in paths_array:
                ptelp = list(path_to_edge_list(path))
                for i in range(num_split_entity):
                    com_path_edges_dict[
                        (k + i * 0.001, source, target, demand / num_split_entity)
                    ] += ptelp

        # for each edge, split all commodities using that edge across subproblems
        subproblem_com_indices = defaultdict(list)
        current_subproblem = 0
        for (u, v) in problem.G.edges:
            coms_on_edge = [
                x
                for x in com_path_edges_dict.keys()
                if (u, v) in com_path_edges_dict[x]
            ]

            # split commodities that share path across all subproblems
            for (k, source, target, demand) in coms_on_edge:
                subproblem_com_indices[current_subproblem] += [
                    (k, source, target, demand)
                ]
                current_subproblem = (current_subproblem + 1) % self._num_subproblems
                # remove commodity from cosideration when processing later edges
                del com_path_edges_dict[(k, source, target, demand)]

        # create subproblems, zero out commodities in traffic matrix that aren't assigned to each
        sub_problems = []
        for i in range(self._num_subproblems):

            sub_problems.append(problem.copy())
            # zero-out the traffic matrices; they will be populated later using entity_assignments_lists
            for u in sub_problems[-1].G.nodes:
                for v in sub_problems[-1].G.nodes:
                    sub_problems[-1].traffic_matrix.tm[u, v] = 0

            # assigned commodity to subproblem i
            for _, source, target, demand in subproblem_com_indices[i]:
                sub_problems[-1].traffic_matrix.tm[source, target] += demand

            # split the capacity of each link
            for u, v in sub_problems[-1].G.edges:
                sub_problems[-1].G[u][v]["capacity"] = (
                    sub_problems[-1].G[u][v]["capacity"] / self._num_subproblems
                )

        return sub_problems
