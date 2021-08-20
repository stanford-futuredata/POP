import copy
import numpy as np
import random
import time


class PartitionedProblem:
    def __init__(self, policy, k):
        self._policy_instances = [copy.deepcopy(policy) for _ in range(k)]
        assert k is not None
        self._k = k
        self._name = policy._name

    @property
    def name(self):
        return self._name

    def get_max_time(self):
        return np.max(np.array(self._times))

    def get_allocation(self, *args, **kwargs):
        args_list = list(args)
        throughputs = args_list[0]
        cluster_spec = args_list[-1]

        sub_problem_cluster_spec = {x: cluster_spec[x] // self._k
                                    for x in cluster_spec}

        job_to_sub_problem_assignment = {}
        job_ids = []
        for job_id in throughputs:
            if not job_id.is_pair():
                job_ids.append(job_id)
        for i, job_id in enumerate(job_ids):
            job_to_sub_problem_assignment[job_id[0]] = \
                random.randint(0, self._k-1)

        sub_problem_throughputs = []
        for i in range(self._k):
            sub_problem_throughputs.append({})
            for job_id in throughputs:
                if (job_to_sub_problem_assignment[job_id[0]] == i) and (
                    not job_id.is_pair() or (job_to_sub_problem_assignment[job_id[1]] == i)):
                    sub_problem_throughputs[-1][job_id] = copy.copy(
                        throughputs[job_id])

        full_allocation = {}
        self._times = []
        for i in range(self._k):
            start_time = time.time()
            args_list_sub_problem = copy.deepcopy(args_list[1:])
            args_list_sub_problem[-1] = sub_problem_cluster_spec
            args_list_sub_problem = [sub_problem_throughputs[i]] + args_list_sub_problem
            sub_problem_allocation = self._policy_instances[i].get_allocation(
                *args_list_sub_problem, **kwargs)
            if sub_problem_allocation is not None:
                for job_id in sub_problem_allocation:
                    full_allocation[job_id] = sub_problem_allocation[job_id]
            self._times.append(time.time() - start_time)

        return full_allocation
