from .abstract_pop_splitter import AbstractPOPSplitter
from .utils import create_edges_onehot_dict, split_generic, compute_precluster
import math


class GenericSplitter(AbstractPOPSplitter):
    # TODO: change this so that it no longer takes in a PathFormulation object as an argument
    def __init__(
        self, num_subproblems, pf, method="means", split_fraction=0.1, verbose=False
    ):
        super().__init__(num_subproblems)
        self._pf = pf
        self.verbose = verbose
        self.method = method
        self.split_fraction = split_fraction

    def split(self, problem):
        input_dict, np_data = create_edges_onehot_dict(
            problem, self._pf, self._num_subproblems, self.split_fraction
        )

        # create subproblems, zero out commodities in traffic matrix that aren't assigned to each
        sub_problems = [problem.copy() for _ in range(self._num_subproblems)]
        if self._num_subproblems == 1:
            return sub_problems

        # zero-out the traffic matrices; they will be populated later using entity_assignments_lists
        for sp in sub_problems:
            for u in sp.G.nodes:
                for v in sp.G.nodes:
                    sp.traffic_matrix.tm[u, v] = 0

        precluster = None
        categorical = None
        if self.method == "cluster":
            categorical = list(range(len(problem.G.edges)))
            precluster = compute_precluster(
                np_data,
                int(math.sqrt(len(problem.G.nodes))),
                categorical_indices=categorical,
            )

        entity_assignments_lists = split_generic(
            input_dict,
            self._num_subproblems,
            verbose=self.verbose,
            method=self.method,
            precluster=precluster,
            np_data=np_data,
            categorical=categorical,
        )

        if self.method == "cluster":
            # replace commodity id index with commodity id (assuming ordereddict)
            commodity_ids_list = list(input_dict.keys())
            for i in range(len(entity_assignments_lists)):
                entity_assignments_lists[i] = [
                    commodity_ids_list[x] for x in entity_assignments_lists[i]
                ]

        for i in range(self._num_subproblems):

            # populate TM for commodities assigned to subproblem i
            for _, source, target, demand in entity_assignments_lists[i]:
                sub_problems[i].traffic_matrix.tm[source, target] += demand

            # split the capacity of each link
            for u, v in sub_problems[i].G.edges:
                sub_problems[i].G[u][v]["capacity"] = (
                    sub_problems[i].G[u][v]["capacity"] / self._num_subproblems
                )

        return sub_problems
