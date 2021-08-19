from .abstract_test import AbstractTest
from ..problems import FeasibilityProblem1
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi


class FeasibilityTest(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = FeasibilityProblem1()

    @property
    def name(self):
        return "feas"

    def run(self):
        ncf = NcfEpi.new_total_flow(4)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 0, 1, 1, 2, 2, 2])
        ncf.solve(self.problem, hc)

        self.assert_eq_epsilon(ncf.obj_val, 2.0)
