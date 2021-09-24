import argparse
import numpy as np
import os
import re


def get_logfile_paths_helper(directory_name):
    logfile_paths = []
    for root, _, file_names in os.walk(directory_name):
        if len(file_names) > 0:
            logfile_paths.extend(
                [os.path.join(root, file_name)
                 for file_name in file_names])
    return logfile_paths


def get_logfile_paths(directory_name, static_trace=False):
    logfile_paths = []
    for logfile_path in get_logfile_paths_helper(directory_name):
        if static_trace:
            m = re.match(
                r'.*v100=(\d+)\.p100=(\d+)\.k80=(\d+)/(.*)/num_sub_problems=(\d+)/'
                 'seed=(\d+)/num_total_jobs=(\d+)\.log', logfile_path)
        else:
            m = re.match(
                r'.*v100=(\d+)\.p100=(\d+)\.k80=(\d+)/(.*)/num_sub_problems=(\d+)/'
                 'seed=(\d+)/lambda=(\d+\.\d+)\.log', logfile_path)
        if m is None: continue
        v100s = int(m.group(1))
        p100s = int(m.group(2))
        k80s = int(m.group(3))
        policy = m.group(4)
        num_sub_problems = int(m.group(5))
        seed = int(m.group(6))
        lambda_or_num_total_jobs = float(m.group(7))
        logfile_paths.append((v100s, p100s, k80s, policy, num_sub_problems, seed,
                              lambda_or_num_total_jobs, logfile_path))
    return logfile_paths


def average_jct_fn(logfile_path, min_job_id=None, max_job_id=None):
    job_completion_times = []
    with open(logfile_path, 'r') as f:
        lines = f.readlines()
        for line in lines[-10000:]:
            m = re.match(r'Job (\d+): (\d+\.\d+)', line)
            if m is not None:
                job_id = int(m.group(1))
                job_completion_time = round(float(m.group(2)), 3)
                if min_job_id is None or min_job_id <= job_id:
                    if max_job_id is None or job_id <= max_job_id:
                        job_completion_times.append(
                            job_completion_time)
    if len(job_completion_times) == 0:
        return None
    return round(np.mean(job_completion_times) / 3600, 3)


def runtime_fn(logfile_path):
    runtime = None
    with open(logfile_path, 'r') as f:
        lines = f.readlines()
        for line in lines[-10:]:
            # Mean allocation computation time: 0.0105 seconds
            m = re.match(r'Mean allocation computation time: (\d+\.\d+) seconds', line)
            if m is not None:
                runtime = round(float(m.group(1)), 3)
    return runtime


def print_all_results_in_directory(logfile_directory):
    print("V100s\tP100s\tK80s\tPolicy\t\t\tK\tSeed\tLambda\tMetric\tRuntime")
    logfile_paths = get_logfile_paths(logfile_directory)

    for logfile_path in logfile_paths:
        (v100s, p100s, k80s, policy, num_sub_problems, seed,
         lambda_or_num_total_jobs, logfile_path) = logfile_path
        average_jct = average_jct_fn(logfile_path)
        runtime = runtime_fn(logfile_path)
        results = [v100s, p100s, k80s, policy, num_sub_problems, seed,
                   lambda_or_num_total_jobs, average_jct, runtime]
        print("\t".join([str(x) for x in results]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse output logfiles')
    parser.add_argument('-l', "--logfile-directory", type=str,
                        required=True,
                        help='Directory with output logfiles')

    args = parser.parse_args()
    print_all_results_in_directory(args.logfile_directory)
