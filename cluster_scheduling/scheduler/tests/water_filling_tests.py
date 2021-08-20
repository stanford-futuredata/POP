import sys; sys.path.append("..")
from policies import max_min_fairness_water_filling

import random
import time

import numpy as np
np.set_printoptions(precision=3, suppress=True)

def test_water_filling():
    policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
        priority_reweighting_policies=None)
    worker_types = ['k80', 'p100', 'v100']
    cluster_spec = {worker_type: 64 for worker_type in worker_types}
    num_jobs = 300
    print("Total number of jobs: %d" % num_jobs)
    unflattened_throughputs = {}
    scale_factors = {}
    unflattened_priority_weights = {}
    num_workers_requested = 0
    for i in range(num_jobs):
        throughputs = [random.random() for i in range(len(worker_types))]
        throughputs.sort()
        unflattened_throughputs[i] = {
            worker_types[i]: throughputs[i] for i in range(len(worker_types))}
        scale_factors[i] = 2 ** random.randint(0, 2)
        num_workers_requested += scale_factors[i]
        unflattened_priority_weights[i] = random.randint(1, 5)
        print("Job %d: Throughputs=%s, Priority=%d, Scale factor=%d" % (
            i, unflattened_throughputs[i], unflattened_priority_weights[i],
            scale_factors[i]))
    print("Total number of workers requested: %d" % num_workers_requested)
    start_time = time.time()
    allocation = policy.get_allocation(unflattened_throughputs, scale_factors,
                                       unflattened_priority_weights,
                                       cluster_spec, verbose=True)
    print()
    return time.time() - start_time


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    times = []
    for i in range(5):
        times.append(test_water_filling())
    print("Average time per problem: %.2f seconds" % np.mean(np.array(times)))
