from .abstract_test import AbstractTest
from ..problems import FlowPathConstruction
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi

# This test case is useful for testing how we construct
# flows per path per commod using NCFlow


class FlowPathConstructionTest(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = FlowPathConstruction()

    @property
    def name(self):
        return "flow-path-construction"

    def run(self):
        ncf = NcfEpi.new_total_flow(4)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 1, 1, 2, 2])
        ncf.solve(self.problem, hc)

        self.assert_feasibility(ncf)
