import sys; sys.path.append("..")
from job_id_pair import JobIdPair
from policies import allox, finish_time_fairness, gandiva, isolated, \
    max_min_fairness, max_min_fairness_water_filling, max_sum_throughput
from policies import partitioned_problem

import itertools
import numpy as np
import unittest

class TestPolicies(unittest.TestCase):

    def test_allox(self):
        policy = allox.AlloXPolicy()
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            2: {'v100': 4.0, 'p100': 3.0, 'k80': 2.0},
            3: {'v100': 1.0, 'p100': 1.0, 'k80': 1.0}
        }
        scale_factors = {0: 1, 1: 1, 2: 1, 3: 1}
        times_since_start = {0: 0, 1: 0, 2: 0, 3: 0}
        num_steps_remaining = {0: 300, 1: 500, 2: 1000, 3: 500}
        cluster_spec = {
            'v100': 2,
            'p100': 1,
            'k80': 3
        }
        allocation1 = policy.get_allocation(unflattened_throughputs,
                                            scale_factors,
                                            times_since_start,
                                            num_steps_remaining,
                                            cluster_spec)

        unflattened_throughputs = {
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            2: {'v100': 4.0, 'p100': 3.0, 'k80': 2.0},
            3: {'v100': 1.0, 'p100': 1.0, 'k80': 1.0},
            4: {'v100': 4.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {1: 1, 2: 1, 3: 1, 4: 1}
        times_since_start = {1: 200, 2: 200, 3: 200, 4: 0}
        num_steps_remaining = {1: 100, 2: 800, 3: 400, 4: 100}
        allocation2 = policy.get_allocation(unflattened_throughputs,
                                            scale_factors,
                                            times_since_start,
                                            num_steps_remaining,
                                            cluster_spec)

        # Jobs still running should have an unchanged allocation.
        for job_id in allocation1:
            if job_id in allocation2:
                assert(allocation1[job_id] == allocation2[job_id])

        # Number of workers used should be less than the total number of
        # available workers.
        for allocation in [allocation1, allocation2]:
            num_workers_used = {worker_type: 0 for worker_type in cluster_spec}
            for job_id in allocation:
                for worker_type in cluster_spec:
                    num_workers_used[worker_type] += allocation[job_id][worker_type]
            for worker_type in cluster_spec:
                assert(num_workers_used[worker_type] <= cluster_spec[worker_type])

    def test_isolated(self):
        isolated_policy = isolated.IsolatedPolicy()
        max_min_fairness_policy = max_min_fairness.MaxMinFairnessPolicy(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        unflattened_priority_weights = {0: 1, 1: 1}
        all_scale_factors = [
            {0: 1, 1: 1}, {0: 2, 1: 1}, {0: 2, 1: 2}, {0:4, 1:1}
        ]
        all_cluster_specs = [
            {'v100': 4, 'p100': 4, 'k80': 4},
            {'v100': 4, 'p100': 8, 'k80': 6}
        ]
        for scale_factors, cluster_spec in itertools.product(
            all_scale_factors, all_cluster_specs):
            isolated_allocation = isolated_policy.get_allocation(
                unflattened_throughputs,
                scale_factors,
                cluster_spec)
            max_min_fairness_allocation = max_min_fairness_policy.get_allocation(
                unflattened_throughputs, scale_factors,
                unflattened_priority_weights, cluster_spec)
            isolated_objectives = []
            max_min_fairness_objectives = []
            for job_id in unflattened_throughputs:
                isolated_objective = 0.0
                max_min_fairness_objective = 0.0
                for worker_type in cluster_spec:
                    isolated_objective += (
                        isolated_allocation[job_id][worker_type] *
                        scale_factors[job_id])
                    max_min_fairness_objective += (
                        max_min_fairness_allocation[job_id][worker_type] *
                        scale_factors[job_id])
                isolated_objectives.append(isolated_objective)
                max_min_fairness_objectives.append(max_min_fairness_objective)
            assert(np.isclose(min(isolated_objectives),
                              min(max_min_fairness_objectives)))

    def test_gandiva(self):
        policy = gandiva.GandivaPolicy()
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec)

        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0},
            JobIdPair(1, None): {'v100': 3.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0)},
        }
        cluster_spec = {
            'v100': 1,
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec)

        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0},
            JobIdPair(1, None): {'v100': 3.0},
            JobIdPair(0, 1): {'v100': (1.0, 1.0)},
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec)

    def test_max_min_fairness_with_perf(self):
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        unflattened_priority_weights = {0: 1, 1: 1}
        cluster_spec = {
            'v100': 1,
            'p100': 2,
            'k80': 3
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              cluster_spec)

    def test_partitioned_max_min_fairness_with_perf(self):
        policy = partitioned_problem.PartitionedProblem(
            max_min_fairness.MaxMinFairnessPolicyWithPerf(solver='ECOS'), 2)
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(2, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(3, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1,
            JobIdPair(2, None): 1,
            JobIdPair(3, None): 1,
        }
        unflattened_priority_weights = {
            JobIdPair(0, None): 1, JobIdPair(1, None): 1,
            JobIdPair(2, None): 1, JobIdPair(3, None): 1}
        cluster_spec = {
            'v100': 2,
            'p100': 4,
            'k80': 6
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              cluster_spec)

    def test_max_min_fairness_with_packing(self):
        policy = max_min_fairness.MaxMinFairnessPolicyWithPacking(
            solver='ECOS')
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        unflattened_priority_weights = {JobIdPair(0, None): 1,
                                        JobIdPair(1, None): 1}
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              cluster_spec)

    def test_max_min_fairness_water_filling_with_packing(self):
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPacking()
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        unflattened_priority_weights = {JobIdPair(0, None): 1,
                                        JobIdPair(1, None): 1}
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              cluster_spec)

    def test_max_min_fairness_with_packing_using_job_type_throughputs(self):
        policy = max_min_fairness.MaxMinFairnessPolicyWithPacking(
            solver='ECOS')
        unflattened_job_type_throughputs = {
            ('A', 1): {
                'v100': {
                    None: 2.0,
                    ('A', 1): 0.0,
                    ('B', 1): 1.0,
                },
                'p100': {
                    None: 1.0,
                    ('A', 1): 0.0,
                    ('B', 1): 0.5,
                },
                'k80': {
                    None: 0.5,
                    ('A', 1): 0.0,
                    ('B', 1): 0.25,
                },
            },
            ('B', 1): {
                'v100': {
                    None: 10.0,
                    ('A', 1): 5.0,
                    ('B', 1): 0.0,
                },
                'p100': {
                    None: 5.0,
                    ('A', 1): 2.5,
                    ('B', 1): 0.0,
                },
                'k80': {
                    None: 2.5,
                    ('A', 1): 1.25,
                    ('B', 1): 0.0
                },
            },
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        unflattened_priority_weights = {JobIdPair(0, None): 1,
                                        JobIdPair(1, None): 1}
        job_id_to_job_type = {
            JobIdPair(0, None): ('A', 1),
            JobIdPair(1, None): ('B', 1),
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation_using_job_type_throughputs(
                unflattened_job_type_throughputs, job_id_to_job_type,
                scale_factors, unflattened_priority_weights, cluster_spec)

    def test_finish_time_fairness(self):
        policy = finish_time_fairness.FinishTimeFairnessPolicy(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        unflattened_priority_weights = {0: 1, 1: 1}
        times_since_start = {0: 0, 1: 0}
        num_steps_remaining = {0: 300, 1: 500}
        cluster_spec = {
            'v100': 1,
            'p100': 2,
            'k80': 3
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              times_since_start, num_steps_remaining,
                              cluster_spec)
        num_steps_remaining = {0: 200, 1: 300}
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              times_since_start, num_steps_remaining,
                              cluster_spec)

    def test_finish_time_fairness_with_perf(self):
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPerf(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        unflattened_priority_weights = {0: 1, 1: 1}
        times_since_start = {0: 0, 1: 0}
        num_steps_remaining = {0: 300, 1: 500}
        cluster_spec = {
            'v100': 1,
            'p100': 2,
            'k80': 3
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              times_since_start, num_steps_remaining,
                              cluster_spec)
        num_steps_remaining = {0: 200, 1: 300}
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              times_since_start, num_steps_remaining,
                              cluster_spec)

    def test_finish_time_fairness_with_packing(self):
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPacking(
            solver='ECOS')
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        unflattened_priority_weights = {JobIdPair(0, None): 1,
                                        JobIdPair(1, None): 1}
        times_since_start = {JobIdPair(0, None): 0, JobIdPair(1, None): 0}
        num_steps_remaining = {JobIdPair(0, None): 300, JobIdPair(1, None): 500}
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              times_since_start, num_steps_remaining,
                              cluster_spec)
        num_steps_remaining = {JobIdPair(0, None): 200, JobIdPair(1, None): 300}
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              unflattened_priority_weights,
                              times_since_start, num_steps_remaining,
                              cluster_spec)

    def test_throughput_sum(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerf(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs=None)

    def test_throughput_sum_normalized_by_cost(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerf(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs={'v100': 3.1, 'p100': 2.0, 'k80': 0.8})

    def test_throughput_sum_normalized_by_cost_with_SLOs(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerfSLOs(
            solver='ECOS')
        unflattened_throughputs = {
            0: {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            1: {'v100': 3.0, 'p100': 2.0, 'k80': 1.0}
        }
        scale_factors = {
            0: 1,
            1: 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs={'v100': 3.1, 'p100': 2.0, 'k80': 0.8},
                              SLOs={0: 1000}, num_steps_remaining={0: 100})

    def test_throughput_sum_normalized_by_cost_with_packing_and_SLOs(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPackingSLOs(
            solver='ECOS')
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs={'v100': 3.1, 'p100': 2.0, 'k80': 0.8},
                              SLOs={JobIdPair(0, None): 1000},
                              num_steps_remaining={JobIdPair(0, None): 100})

    def test_throughput_sum_normalized_by_cost_with_packing_and_SLOs_v2(self):
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPackingSLOs(
            solver='ECOS')
        unflattened_throughputs = {
            JobIdPair(0, None): {'v100': 2.0, 'p100': 1.0, 'k80': 0.5},
            JobIdPair(1, None): {'v100': 3.0, 'p100': 2.0, 'k80': 1.0},
            JobIdPair(0, 1): {'v100': (2.0, 3.0), 'p100': (1.0, 2.0),
                              'k80': (0.5, 1.0)},
        }
        scale_factors = {
            JobIdPair(0, None): 1,
            JobIdPair(1, None): 1
        }
        cluster_spec = {
            'v100': 1,
            'p100': 1,
            'k80': 1
        }
        policy.get_allocation(unflattened_throughputs, scale_factors,
                              cluster_spec,
                              instance_costs={'v100': 1.0, 'p100': 1.0, 'k80': 1.0},
                              SLOs={JobIdPair(0, None): 1000},
                              num_steps_remaining={JobIdPair(0, None): 100})


if __name__=='__main__':
    unittest.main()
