from .abstract_partitioning_method import AbstractPartitioningMethod
import numpy as np
from networkx.algorithms import community


# Partition based on community-finding algorithms implemented in NetworkX:
# https://networkx.github.io/documentation/stable/reference/algorithms/community.html
class NetworkXPartitioning(AbstractPartitioningMethod):
    def __init__(self, part_fn_str, num_partitions=None, seed=0):
        super().__init__(num_partitions=num_partitions, weighted=False)
        self.set_partition_fn(part_fn_str)
        self.seed = seed

    def asyn_lpa(self, prob):
        return community.asyn_lpa_communities(prob.G, weight="capacity", seed=self.seed)

    def set_partition_fn(self, part_fn_str):
        if part_fn_str == "label_propagation":
            self._part_fn = self.asyn_lpa
        else:
            raise Exception(
                "{} not a valid NetworkX partition function".format(part_fn_str)
            )

    def _partition_impl(self, problem):
        G = problem.G
        if not hasattr(self, "_num_partitions"):
            self._num_partitions = self._default_num_partitions(G)

        p_v = np.zeros(len(problem.G.nodes), dtype=np.int32)
        for part_id, part in enumerate(self._part_fn(problem)):
            p_v[list(part)] = part_id
        self._num_partitions = len(np.unique(p_v))
        return p_v
