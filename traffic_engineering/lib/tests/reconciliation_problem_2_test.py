from .abstract_test import AbstractTest
from ..problems import ReconciliationProblem2
from ..partitioning.hard_coded_partitioning import HardCodedPartitioning
from ..algorithms.ncflow.ncflow_edge_per_iter import NCFlowEdgePerIter as NcfEpi


class ReconciliationProblem2Test(AbstractTest):
    def __init__(self):
        super().__init__()
        self.problem = ReconciliationProblem2()

    @property
    def name(self):
        return "recon2"

    def run(self):
        ncf = NcfEpi.new_total_flow(4)
        hc = HardCodedPartitioning(partition_vector=[0, 0, 1, 1])
        ncf.solve(self.problem, hc)

        self.assert_feasibility(ncf)

        self.assert_eq_epsilon(ncf.r1_obj_val, 40.0)
        self.assert_eq_epsilon(ncf.intra_obj_vals[0], 0.0)
        self.assert_eq_epsilon(ncf.intra_obj_vals[1], 0.0)
        self.assert_eq_epsilon(ncf.r3_obj_val, 10.0)
        self.assert_eq_epsilon(ncf.obj_val, 10.0)
