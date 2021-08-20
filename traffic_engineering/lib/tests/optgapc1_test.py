from .abstract_test import AbstractTest
from ..problems import OptGapC1
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi

# This test illustrates the optimality gap from relaxing condition C1
# That is, that all the demands be satisfiable in order for optimality to hold.
# Here, a flow has multiple bottlenecks in different meta-nodes and
# flows that share only one of those bottlenecks lose flow; leading to optimality gap.


class OptGapC1Test(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = OptGapC1()

    @property
    def name(self):
        return "optgapc1"

    def run(self):
        ncf = NcfEpi.new_total_flow(4)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 0, 1, 1, 1])
        ncf.solve(self.problem, hc)

        self.assert_feasibility(ncf)

        # this is a shame; the optimal solution here should be 8; we get 1.0
        self.assert_geq_epsilon(ncf.obj_val, 5.0)
        self.assert_leq_epsilon(ncf.obj_val, 7.0)
