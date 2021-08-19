from .abstract_test import AbstractTest
from ..problems import SingleEdgeB
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi


class SingleEdgeBTest(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = SingleEdgeB()

    @property
    def name(self):
        return "SingleEdgeB"

    def run(self):
        ncf = NcfEpi.new_total_flow(4, verbose=True)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 1, 1, 1, 1, 2, 2])
        ncf.solve(self.problem, hc)
        print(ncf.obj_val)

        self.assert_feasibility(ncf)
