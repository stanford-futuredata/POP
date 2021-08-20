from .abstract_test import AbstractTest
from ..problems import OptGapC2
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi

# This test illustrates the optimality gap from relaxing condition C2
# That is, that there is no undirected cycle among meta-nodes
# Here, a flow has two paths but R1 picks one and only in R2 is the
# bottleneck discovered leading to lost flow


class OptGapC2Test(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = OptGapC2()

    @property
    def name(self):
        return "optgapc2"

    def run(self):
        ncf = NcfEpi.new_total_flow(4)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 1, 1, 2, 2, 3, 3])
        ncf.solve(self.problem, hc)

        self.assert_feasibility(ncf)

        # this is a shame; the optimal solution here should be 8; we get 1.0
        self.assert_geq_epsilon(ncf.obj_val, 3.0)
        self.assert_leq_epsilon(ncf.obj_val, 8.0)
