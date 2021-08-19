from ..abstract_formulation import AbstractFormulation
from ...runtime_utils import parallelized_rt


class NCFlowAbstract(AbstractFormulation):
    @property
    def runtime(self):
        return self.runtime_est(14)  # Hardcoded for GCR machines

    def runtime_est(self, num_threads, breakdown=False):

        rts = self._runtime_dict
        r2_time = parallelized_rt(list(rts["r2"].values()), num_threads)
        reconciliation_time = parallelized_rt(
            list(rts["reconciliation"].values()), num_threads
        )

        if "kirchoffs" in rts:
            kirchoffs_time = parallelized_rt(
                list(rts["kirchoffs"].values()), num_threads
            )
        else:
            kirchoffs_time = 0

        print(
            "Runtime breakdown: R1 {} R2// {} Recon// {} R3 {} Kirchoffs// {} #threads {}".format(
                rts["r1"],
                r2_time,
                reconciliation_time,
                rts["r3"],
                kirchoffs_time,
                num_threads,
            )
        )
        if breakdown:
            return rts["r1"], r2_time, reconciliation_time, rts["r3"], kirchoffs_time

        total_time = (
            rts["r1"] + r2_time + reconciliation_time + rts["r3"] + kirchoffs_time
        )

        return total_time
