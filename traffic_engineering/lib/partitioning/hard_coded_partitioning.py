from .abstract_partitioning_method import AbstractPartitioningMethod
import numpy as np


class HardCodedPartitioning(AbstractPartitioningMethod):
    def __init__(self, partition_vector):
        if not isinstance(partition_vector, np.ndarray):
            _partition_vector = np.array(partition_vector)
        else:
            _partition_vector = partition_vector

        super().__init__(num_partitions=max(partition_vector) + 1, weighted=False)
        self._use_cache = False
        self._partition_vector = _partition_vector

    def _partition_impl(self, problem):
        assert len(self.partition_vector) == len(problem.G.nodes)
        return self.partition_vector
