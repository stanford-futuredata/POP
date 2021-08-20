# /lfs/1/deepak/logs/pop_test/raw_logs/v100\=32.p100\=32.k80\=32/max_min_fairness_perf/num_sub_problems\=1/seed\=0/lambda\=600.000000.log
import re


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
        logfile_paths.append((v100s, p100s, k80s, policy, seed,
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
                job_completion_time = float(m.group(2))
                if min_job_id is None or min_job_id <= job_id:
                    if max_job_id is None or job_id <= max_job_id:
                        job_completion_times.append(
                            job_completion_time)
    if len(job_completion_times) == 0:
        return None
    return np.mean(job_completion_times) / 3600


if __name__ == '__main__':
    logfile_directory = '/lfs/1/deepak/logs/pop_test/'
    logfile_paths = get_logfile_paths(logfile_directory)

    for logfile_path in logfile_paths:
        (v100s, p100s, k80s, policy, seed, lambda_or_num_total_jobs, logfile_path) = logfile_path
        average_jct = average_jct_fn(logfile_path)
        print(v100s, p100s, k80s, policy, seed, lambda_or_num_total_jobs, logfile_path, average_jct)