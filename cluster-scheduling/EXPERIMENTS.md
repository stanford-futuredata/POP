# OSDI 2020 Experiments

This document describes how to run the main experiments in the OSDI 2020 paper.

## Setup

### Software Dependencies

Gavel is implemented in Python. We have tested Gavel's simulator on Ubuntu 16.04
with Python 3.8; this can be installed using
[Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Required software dependencies can be installed using,

```bash
apt-get -y install cmake g++ gcc libnuma-dev make numactl zlib1g-dev
pip install -r scheduler/requirements.txt
cd scheduler; make
```

These software dependencies have already been installed on the following
AMI on Amazon EC2,

| Field  | Value |
| -------------  | ------------- |
| Cloud Provider | AWS |
| Region         | us-east-1  |
| AMI ID         | ami-03e41a79bb745ce18  |
| AMI Name       | gavel |

See [this link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami.html)
for how to find and launch a public AMI (this assumes you have a valid billable AWS account setup).


### Data for Physical Cluster Experiments

Data for the workloads used in the physical cluster experiments are
available in the `gavel` AMI available on Amazon EC2.

## Reproducing Experiments

Gavel's heterogeneity-aware policies and scheduling mechanism can be evaluated
either in simulation or on a physical cluster.

The evaluation in the paper largely shows results on a simulated cluster.
We provide a JSON file with measured throughputs for the target workloads
used in our experiments at `simulation_throughputs.json`. Experiments
are run from the `scheduler/` sub-directory.

We also include instructions on how to deploy Gavel on a small physical
cluster.

### Figure 8: Least Attained Service Policy on Continuous-Single Trace

To reproduce Figure 8 in the paper (that is, evaluate variants of the LAS
policy (`max_min_fairness*`) in simulation), one can use the following command
line (this sweep script runs the different policies for multiple traces,
generated using different seeds and Poisson arrival rates on a cluster with
36 V100 GPUs, 36 P100 GPUs, and 36 K80 GPUs):

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p allox gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 6.0 -n 16
```

The output of this script looks like this:

```bash
>> python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l test_logs -j 6 -p allox gandiva max_min_fairness max_min_fairness_perf --seeds 42 1234 15 -c 36:36:36 -a 0.0 -b 6.0 -n 16
[2020-09-03 17:17:49.260052] Running 180 total experiment(s)...
[2020-09-03 17:17:49.535227] [Experiment ID:  0] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=42, lam=9000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.600210] [Experiment ID:  1] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=1234, lam=9000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.699429] [Experiment ID:  2] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=15, lam=9000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.727137] [Experiment ID:  4] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=1234, lam=4500.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.826323] [Experiment ID:  3] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=42, lam=4500.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:17:49.875449] [Experiment ID:  5] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=15, lam=4500.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:21:27.936718] [Experiment ID:  3] Results: average JCT=59770.441018, utilization=0.121262
[2020-09-03 17:21:28.072957] [Experiment ID:  6] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=42, lam=3000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:21:30.507841] [Experiment ID:  4] Results: average JCT=64695.450528, utilization=0.123312
[2020-09-03 17:21:30.639389] [Experiment ID:  7] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=1234, lam=3000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:21:31.027912] [Experiment ID:  5] Results: average JCT=55072.401336, utilization=0.121365
[2020-09-03 17:21:31.161566] [Experiment ID:  8] Configuration: cluster_spec=v100:36|p100:36|k80:36, policy=AlloX_Perf, seed=15, lam=3000.000000, profiling_percentage=1.000000, num_reference_models=26
[2020-09-03 17:21:54.980905] [Experiment ID:  2] Results: average JCT=54834.129989, utilization=0.060839
[2020-09-03 17:22:00.203122] [Experiment ID:  1] Results: average JCT=64494.582453, utilization=0.061878
[2020-09-03 17:22:01.456350] [Experiment ID:  0] Results: average JCT=59492.106756, utilization=0.060797
...
```

This script can take on the order of hours to complete for most of the datapoints,
and a couple of days for the tail; we suggest using `tmux`. The
`max_min_fairness_packed` policy is particularly slow, and can be ommited to
start to get results quickly. There are a couple of different ways to obtain results quicker:
a) use a smaller cluster (e.g., 16 GPUs of each type, controlled by the `-c`
argument), and consequently with a smaller range of input job rates (controlled
by `-a` and `-b` command line arguments), b) use a fewer number of seeds,
c) sweep fewer input job rates (controlled by the `-n` argument),
d) adjust the `-s` and `-e` arguments, which control the
size of the trace, and the size of the set of jobs of interest (e.g., `-s 2000 -e 3000`).

NOTE: Do not let this script run for an unbounded amount of time: traces with really
high input rates will never finish, since the input rate is higher than the average
rate the cluster can complete jobs. As a result, average job completion time for these
traces is infinity (jobs never complete).

Some policies might need to be run for higher input job rates as well. Our
experiments were run using seeds 0, 1, and 2; results with other seeds should
look similar (and we encourage using different seeds).

`scheduler/notebooks/figures/evaluation/continuous_jobs.ipynb` contains
code to parse the resulting logs and produce graphs (can be run using `jupyter
notebook`). The notebook should use the appropriate `log_directory` used
in the above command line.

Notebooks can be used to view the results of in-progress runs -- the sweep
script can be killed once sufficient points are run.
The [`prune_logs.py`](scheduler/notebooks/figures/evaluation/prune_logs.py) script
can be used to clean up really large logfiles that have not completed -- this will
speed up the plotting scripts.

### Figure 9: Least Attained Service Policy on Continuous-Multiple Trace

To reproduce Figure 9, one can use the following command line (use a different
log directory for each figure):

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 3.0 -n 11 --generate-multi-gpu-jobs
```

`scheduler/notebooks/figures/evaluation/continuous_jobs_multigpu.ipynb` contains
relevant parsing and plotting code. As before, `max_min_fairness_packed` takes
the longest time and can be omitted to obtain results quicker.

### Figure 10: Finish Time Fairness Policy on Continuous-Multiple Trace

To reproduce Figure 10, one can use the following command line:

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p finish_time_fairness finish_time_fairness_perf --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 3.0 -n 11 --generate-multi-gpu-jobs
```

Relevant parsing and plotting code is in the
`scheduler/notebooks/figures/evaluation/continuous_jobs_multigpu.ipynb` notebook.

### Makespan Policy on Static-Multiple Trace

To reproduce the makespan policy results:

```bash
python -u scripts/sweeps/run_sweep_static.py -l /path/to/log/directory -j 24 -p gandiva min_total_duration min_total_duration_perf min_total_duration_packed fifo gandiva --seeds 0 1 2 -c 36:36:36 -a 0 -b 500 -n 6 --generate-multi-gpu-jobs
```

Parsing and plotting code is in the
`scheduler/notebooks/figures/evaluation/makespan.ipynb` notebook.

### Figure 11: Multi-Level Fairness Policy

The code for the simulation shown in Figure 11 is in `scheduler/notebooks/figures/evaluation/hierarchical.ipynb`.

### Figure 12: Policy Runtime Scaling

Policy runtimes can be measured using the following command:

```bash
python scripts/microbenchmarks/sweep_policy_runtimes.py -n 32 64 128 256 512 1024 2048 -p max_min_fairness_perf max_min_fairness_packed max_min_fairness_water_filling max_min_fairness_water_filling_packed --num_trials 3
```

This script prints some comma-separated values to `stdout` -- these values should
be copied into a file for the plotting script to work.

`scheduler/notebooks/figures/evaluation/policy_runtimes.ipynb` contains relevant
parsing and plotting code.

### Figure 13: Efficacy of Scheduling Mechanism

The time proportions returned by the policy can be used directly to grant jobs
times on each resource type between "reset events" -- this is a useful comparison
for our scheduling mechanism. This "ideal" scheduling mechanism can be run
for a given policy and trace by appending the `--ideal` argument to any of the
sweep commands above (Figures 9 or 10) -- in our experiment, we used the `max_min_fairness` policy.

Concretely, one can run:
```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p allox gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 6.0 -n 16 --ideal
```

The round durations used by the scheduling mechanism can be similarly studied
by using the `-i` argument -- the default used is 360 seconds, but other round
durations can be used as well. For example, to use 720 seconds:

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p allox gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 6.0 -n 16 -i 720
```


### Physical Cluster

In a physical cluster, Gavel is comprised of a centralized scheduler (deployed on a separate scheduling
server) and workers (each worker has 1 or more GPUs). Jobs are submitted
to the scheduler. The scheduler then computes a heterogeneity-aware allocation for
each active job using its policy framework. It then uses its round-based scheduling
mechanism to determine how to grant resources to jobs.

We provide scripts to launch both the scheduler and workers. Make sure software
dependencies are installed on the scheduler and workers, as described above.
To launch the scheduler, use `scripts/drivers/run_scheduler.py` as follows:
```bash
python scripts/drivers/run_scheduler_with_trace.py \
  --trace traces/physical_cluster/artifact_evaluation.trace \
  --seed 0 \
  --solver ECOS \
  --throughputs_file physical_cluster_throughputs.json \
  --time_per_iteration 360 \
  --policy min_total_duration_perf \
  --expected_num_workers 6
```
Running this command will start the scheduler and log the IP address
the server is running with. Using this IP address, we can launch a worker
as follows:
```bash
python worker.py \
  -t [WORKER_TYPE] \
  -i [IP_ADDRESS] -s 50060 -w 50061 -g [NUM_GPUS_ON_WORKER] \
  --run_dir /path/to/gavel/workloads/pytorch \
  --data_dir /path/to/data \
  --checkpoint_dir /path/to/checkpoints
```
This should be done for all workers in the cluster - for V100 GPUs use 'v100' as the
worker type, and for K80 GPUs use 'k80'.

The included trace for artifact evaluation should complete in about 2 hours using
a cluster size of 6 GPUs (2 V100s and 4 K80s), and is intended merely to show that the infrastructure
required to deploy Gavel in a physical cluster is included; actually replicating
the physical cluster experiments shown in the paper will require more
resources for a much longer duration.

### Measuring Overhead of Running Jobs with Gavel's Preemptive Scheduler

The following commands can be used to measure the overhead of Gavel's
preemptive scheduler. These require only a single GPU.

The first script sweeps through all model types and runs each model
for 10 rounds in configurations with and without lease extensions. It saves
job timelines into the specified directory.
```bash
python scripts/microbenchmarks/sweep_models_for_overhead.py \
  --timeline_dir /path/to/timelines \
  --run_dir /path/to/gavel/workloads/pytorch \
  --data_dir /path/to/data \
  --checkpoint_dir /path/to/checkpoints
```

The next script parses the saved timelines and computes the overhead.
```bash
python scripts/utils/get_job_overhead.py \
  --timeline_dir /path/to/timelines
```
