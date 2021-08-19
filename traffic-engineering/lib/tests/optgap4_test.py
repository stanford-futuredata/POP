from .abstract_test import AbstractTest
from ..problems import OptGap4
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi

# This test case illustrates the need for reconciliation to allow traffic to "criss-cross"
# between nodes that are in neighboring meta-nodes.


class OptGap4Test(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = OptGap4()

    @property
    def name(self):
        return "optgap4"

    def run(self):
        ncf = NcfEpi.new_total_flow(4)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 0, 1, 1, 1])
        ncf.solve(self.problem, hc)

        self.assert_feasibility(ncf)

        # this is a shame; the optimal solution here should be 8; we get 1.0
        self.assert_geq_epsilon(ncf.obj_val, 2.0)
        self.assert_leq_epsilon(ncf.obj_val, 2.0)
