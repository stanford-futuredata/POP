import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import contextlib
import numpy as np
import random
import time

import utils
from job import Job
from job_id_pair import JobIdPair
from job_table import JobTable

def generate_input(num_active_jobs,
                   cluster_spec,
                   policy_name,
                   oracle_throughputs,
                   generate_multi_gpu_jobs,
                   generate_multi_priority_jobs,
                   seed):
    rng = random.Random()
    rng.seed(seed)
    throughputs = {}
    jobs = {}
    for i in range(num_active_jobs):
        job_id = JobIdPair(i, None)
        jobs[i] = utils.generate_job(throughputs=oracle_throughputs,
                                     rng=rng,
                                     generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                                     generate_multi_priority_jobs=generate_multi_priority_jobs)
        job_type_key = (jobs[i].job_type, jobs[i].scale_factor)
        throughputs[job_id] = {}
        for worker_type in cluster_spec:
            throughputs[job_id][worker_type] = \
                oracle_throughputs[worker_type][job_type_key]['null']
    if 'pack' in policy_name:
        for i in range(num_active_jobs): 
            job_type_key = (jobs[i].job_type, jobs[i].scale_factor)
            for j in range(num_active_jobs):
                if i < j and jobs[i].scale_factor == jobs[j].scale_factor:
                    other_job_type_key = \
                        (jobs[j].job_type, jobs[j].scale_factor)
                    throughputs[JobIdPair(i, j)] = {}
                    for worker_type in cluster_spec:
                        throughputs[JobIdPair(i, j)][worker_type] = \
                            oracle_throughputs[worker_type][job_type_key][other_job_type_key]
    scale_factors = {
        JobIdPair(i, None): jobs[i].scale_factor for i in range(num_active_jobs)
    }
    return throughputs, jobs, scale_factors

def measure_runtime(num_active_jobs, policy_name,
                    oracle_throughputs, generate_multi_gpu_jobs,
                    generate_multi_priority_jobs, num_trials, solver):
    cluster_spec = {
        'v100': num_active_jobs // 4,
        'p100': num_active_jobs // 4,
        'k80': num_active_jobs // 4,
    }
    print(cluster_spec)

    results_str = '%s,%d' % (policy_name, num_active_jobs)
    results = []
    for trial in range(num_trials):
        throughputs, jobs, scale_factors = generate_input(
            num_active_jobs, cluster_spec,
            policy_name, oracle_throughputs,
            generate_multi_gpu_jobs,
            generate_multi_priority_jobs, seed=trial+2)
        if "water_filling" in policy_name:
            num_entities = 5
            priority_reweighting_policies = {}
            entity_to_job_mapping = {}
            entity_weights = {}
            for i in range(num_entities):
                entity_id = 'entity%d' % i
                priority_reweighting_policies[entity_id] = 'fairness'
                entity_to_job_mapping[entity_id] = []
                entity_weights[entity_id] = random.randint(1, 3)
            policy = utils.get_policy(policy_name, solver=solver,
                priority_reweighting_policies=priority_reweighting_policies)
        else:
            policy = utils.get_policy(policy_name, solver=solver)
        start_time = time.time()
        with open('/dev/null', 'w') as f:
            with contextlib.redirect_stdout(f):
                if policy.name.startswith('MaxMinFairness'):
                    priority_weights = {
                        JobIdPair(i, None): jobs[i].priority_weight for i in range(num_active_jobs)
                    }
                    if "WaterFilling" in policy.name:
                        for i in range(num_active_jobs):
                            entity_id = 'entity%d' % random.randint(0, num_entities-1)
                            entity_to_job_mapping[entity_id].append(JobIdPair(i, None))
                        policy.get_allocation(
                            throughputs, scale_factors, priority_weights,
                            cluster_spec,
                            entity_weights=entity_weights,
                            entity_to_job_mapping=entity_to_job_mapping)
                    else:
                        policy.get_allocation(
                            throughputs, scale_factors, priority_weights,
                            cluster_spec)
                elif policy.name.startswith('MinTotalDuration'):
                    num_steps_remaining = {
                        JobIdPair(i, None): jobs[i].num_steps for i in range(num_active_jobs)
                    }
                    policy.get_allocation(
                        throughputs, scale_factors, num_steps_remaining,
                        cluster_spec)
                else:
                    policy.get_allocation(
                        throughputs, scale_factors, cluster_spec)

        runtime = time.time() - start_time
        results.append(runtime)
    for result in results:
        results_str += ',' + str(result)
    results_str += ',' + str(np.mean(results))
    return results_str

def main(args):
    all_num_active_jobs = args.num_active_jobs
    all_policies = args.policies
    oracle_throughputs =\
        utils.read_all_throughputs_json_v2(args.throughputs_file)

    if args.output_file is not None:
        output_file = open(args.output_file, 'w')
    else:
        output_file = None

    header_str = 'Policy,# Jobs'
    for i in range(args.num_trials):
        header_str += ',Trial %d' % (i+1)
    header_str += ',Average'
    if output_file is not None:
        output_file.write('%s\n' % (header_str))
    print(header_str)

    for policy in all_policies:
        for num_active_jobs in all_num_active_jobs:
            results = measure_runtime(num_active_jobs,
                                      policy, oracle_throughputs,
                                      args.generate_multi_gpu_jobs,
                                      args.generate_multi_priority_jobs,
                                      args.num_trials, args.solver)
            if output_file is not None:
                output_file.write('%s\n' % (results))
            print(results)

    if output_file is not None:
        output_file.close

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='')
    parser.add_argument('--throughputs-file', type=str,
                        default='simulation_throughputs.json',
                        help='Oracle throughputs file')
    parser.add_argument('--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('-n', '--num_active_jobs', type=int, nargs='+',
                        default=[2**i for i in range(4, 10)],
                        help='List of number of active jobs to sweep')
    parser.add_argument('-p', '--policies', type=str, nargs='+',
                        default=['max_min_fairness_packed'],
                        help='List of policies to sweep')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of trials to run for each experiment')
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI', 'SCS'],
                        default=None, help='CVXPY solver')
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to output results to')
    args = parser.parse_args()

    main(args)
