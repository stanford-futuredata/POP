from .abstract_test import AbstractTest
from ..problems import WeNeedToFixThis
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi


# Testing WeNeedToFixThis: correct path to target
# isn't visible in source meta-node
class WeNeedToFixThisTest(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = WeNeedToFixThis()

    @property
    def name(self):
        return "we-need-to-fix-this"

    def run(self):
        ncf = NcfEpi.new_total_flow(4)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 1, 1, 1, 1, 2, 2])
        ncf.solve(self.problem, hc)

        self.assert_feasibility(ncf)
