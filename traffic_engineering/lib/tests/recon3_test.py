from .abstract_test import AbstractTest
from ..problems import Recon3
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi


class Recon3Test(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = Recon3()

    @property
    def name(self):
        return "recon3"

    def run(self):
        ncf = NcfEpi.new_total_flow(4)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 0, 1, 1, 1])
        ncf.solve(self.problem, hc)

        self.assert_feasibility(ncf)

        self.assert_eq_epsilon(ncf.obj_val, 5.0)
