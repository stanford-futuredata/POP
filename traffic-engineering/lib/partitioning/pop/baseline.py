from collections import defaultdict
from .abstract_pop_splitter import AbstractPOPSplitter
from ...graph_utils import path_to_edge_list
from math import floor


class BaselineSplitter(AbstractPOPSplitter):
    def __init__(self, num_subproblems):
        super().__init__(num_subproblems)

    def split(self, problem):
        sub_problems = []
        num_rows = len(problem.traffic_matrix.tm)
        rows_per_problem = floor(num_rows / self._num_subproblems)
        shuffled_indices = list(range(num_rows))

        for i in range(self._num_subproblems):

            sub_problems.append(problem.copy())
            for indx, j in enumerate(shuffled_indices):

                # zero out all rows except those in the corresponding block of shuffled indices
                # first, cover special case for last block
                if i == self._num_subproblems - 1:
                    if indx < i * rows_per_problem:
                        sub_problems[-1].traffic_matrix.tm[j, :] = 0

                elif (indx < i * rows_per_problem) or (
                    indx >= (i + 1) * rows_per_problem
                ):
                    sub_problems[-1].traffic_matrix.tm[j, :] = 0

            # split the capacity of each link
            for u, v in sub_problems[-1].G.edges:
                sub_problems[-1].G[u][v]["capacity"] = (
                    sub_problems[-1].G[u][v]["capacity"] / self._num_subproblems
                )

        return sub_problems
