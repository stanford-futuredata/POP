# SOSP 2021 Experiments

This document describes how to run the experiments in the SOSP 2021 paper. These experiments were benchmarked on an `m5.8xlarge` AWS EC2 instance.

## Setup

We have created an image on Amazon EC2 with all software dependencies already
installed. Skip to [the next section](#reproducing-experiments) if using this.

| Field  | Value |
| -------------  | ------------- |
| Cloud Provider | AWS |
| Region         | us-west-2  |
| AMI ID         | ami-025c68f3633e5859e  |
| AMI Name       | pop |

See [this link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami.html)
for how to find and launch a public AMI (this assumes you have a valid billable AWS account setup).

### Software Dependencies

This setup has been verified on Ubuntu 16.04.

1. Install `apt-get` dependencies:
  ```bash
  sudo add-apt-repository ppa:openjdk-r/ppa
  sudo apt-get update && sudo apt -y upgrade
  sudo apt-get install -y build-essential cmake python-dev python3-dev openjdk-11-jdk maven unzip zip htop g++ gcc libnuma-dev make numactl zlib1g-dev
  ```
2. Install [Miniconda with Python 3.8](https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh
```
3. Download and install CPLEX 12.1 free academic version (requires ibm account,
https://www.ibm.com/academic/technology/data-science). Run the installer,
specifying `/home/ubuntu/cplex121` as the install directory. 
4. Download and install [Gurobi 8.1.1](https://packages.gurobi.com/8.1/gurobi8.1.1_linux64.tar.gz):
```bash
wget https://packages.gurobi.com/8.1/gurobi8.1.1_linux64.tar.gz
tar xvf gurobi8.1.1_linux64.tar.gz
```
Add/modify the following environment variables in your `.bashrc`:
```bash
export GUROBI_HOME=$HOME/gurobi811/linux64
export CPLEX_HOME=$HOME/cplex121/cplex
export LD_LIBRARY_PATH=$CPLEX_HOME/bin/x86-64_linux:$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export PATH=$GUROBI_HOME/bin:$PATH
```
Source your `.bashrc` so that these variables are now available.


#### Cluster Scheduling

```bash
cd POP/cluster_scheduling
pip install -r requirements.txt
cd scheduler; make
```

#### Traffic Engineering
```bash
cd POP/traffic_engineering`
conda env create -f environment.yml
conda activate traffic_engineering
pip install -r requirements.txt
./download.sh` # download the traffic matrices used in our experiments.
```

#### Load Balancing

```bash
cd POP/load_balancing
mvn package
```


# Reproducing Experiments

## Gurobi setup
Obtain a [free Gurobi academic
license](https://www.gurobi.com/academia/academic-program-and-licenses/).  Note
that you will have to use `grbgetkey`, which should be available to you once
you've completed the steps in Setup. This will require creating a Gurobi account
and accepting their EULA. After doing so, Gurobi will give you a command to run
to download the key to your machine; for example:
```bash
grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

This will NOT work, because Gurobi requires that the command be run on a
machine that is connected to a university network. To get around this, you
will have to set up SOCKS proxy via `ssh`:
```bash
ssh -D 1337 -f -C -q -N [your_university_username]@[domain_or_public_ip_of_machine_in_university_network]
```
Then, run `grbgetkey` while simultaneously setting `HTTPS_PROXY`:
```bash
HTTPS_PROXY=socks5://localhost:1337 grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

That should work! You can save the Gurobi license file to the `$HOME`
directory: `/home/ubuntu/gurobi.lic`. (You can also now safely kill the
`ssh` proxy process at this point.)

To confirm that the Gurobi license and installation are both setup
correctly, run `gurobi_cl --license`, which should output the path of the license file.
## Figure 6: Max-Min Fairness Policy without Space Sharing

To reproduce Figure 6 in the paper (that is, evaluate the max-min fairness policy presented
in Section XX of the paper), run the following command from `cluster_scheduling/scheduling`
(fill in the output directory as appropriate, this needs to be created beforehand):

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p max_min_fairness_perf --seeds 0 1 2 -c 32:32:32 -a 6.0 -b 6.0 -n 1 --num_sub_problems 1 2 4 8
```

The output of this script looks like this:

```
[2021-08-20 11:44:33.941341] Running 12 total experiment(s)...
[2021-08-20 11:44:34.013608] [Experiment ID:  0] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=0, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=1
[2021-08-20 11:44:34.018696] [Experiment ID:  0] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=1, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=1
[2021-08-20 11:44:34.024131] [Experiment ID:  0] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=2, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=1
[2021-08-20 11:44:34.027469] [Experiment ID:  1] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=0, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=2
[2021-08-20 11:44:34.034641] [Experiment ID:  1] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=1, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=2
[2021-08-20 11:44:34.036817] [Experiment ID:  1] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=2, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=2
[2021-08-20 11:44:34.042734] [Experiment ID:  2] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=0, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=4
[2021-08-20 11:44:34.046593] [Experiment ID:  2] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=1, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=4
[2021-08-20 11:44:34.049944] [Experiment ID:  2] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=2, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=4
[2021-08-20 11:44:34.054563] [Experiment ID:  3] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=0, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=8
[2021-08-20 11:44:34.059386] [Experiment ID:  3] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=1, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=8
[2021-08-20 11:44:34.064934] [Experiment ID:  3] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=2, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=8
...
```

This can take a couple of hours to complete. We suggest using `tmux`.

The generated output logfiles can be analyzed using the postprocessing script
available at `cluster_scheduling/process_logs.py` (the output directory used
above needs to be provided to this script as a command line argument):

```bash
> python process_logs.py -l /path/to/log/directory
V100s	P100s	K80s	Policy			K	Seed	Lambda	Metric
32	32	32	max_min_fairness_perf	2	2	600.0	68.90534538027778
32	32	32	max_min_fairness_perf	2	0	600.0	88.34877724416667
32	32	32	max_min_fairness_perf	4	1	600.0	83.10266556805554
32	32	32	max_min_fairness_perf	4	2	600.0	69.92768312416666
32	32	32	max_min_fairness_perf	4	0	600.0	88.20619143111111
32	32	32	max_min_fairness_perf	1	2	600.0	68.09976061444445
32	32	32	max_min_fairness_perf	1	0	600.0	84.05898847305555
32	32	32	max_min_fairness_perf	8	1	600.0	84.41573162249999
32	32	32	max_min_fairness_perf	8	2	600.0	72.12062279805555
32	32	32	max_min_fairness_perf	8	0	600.0	89.80718732583334
```


## Figure 7: Max-Min Fairness Policy with Space Sharing

To reproduce Figure 7 in the paper, run the following command from `cluster_scheduling/scheduler`:

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p max_min_fairness_packed --seeds 0 1 2 -c 32:32:32 -a 6.4 -b 6.4 -n 1 --num_sub_problems 1 2 4 8
```

Running the space-sharing policies is expensive, so this can take a very long time to run (on the order of a day or more).

## Figure 9: Minimize Makespan

To reproduce Figure 9 in the paper, run the following command from `cluster_scheduling/scheduler`:

```bash
python -u scripts/sweeps/run_sweep_static.py -l /path/to/log/directory -j 24 -p min_total_duration_perf --seeds 0 1 2 -c 32:32:32 -a 700 -b 700 -n 1 --generate-multi-gpu-jobs --num_sub_problems 1 2 4 8
```

## Figure 10: Max-Flow Traffic Engineering, Single Network

To reproduce Figure 10, run the following command from `traffic_engineering/benchmarks`:
```bash
./pop.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --split-fractions 0 --num-subproblems 4 16 64 --split-methods random --obj total_flow

./ncflow.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --obj total_flow

./cspf.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --obj total_flow

./path_form.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --obj total_flow
```

This will create CSV files containing the results (including throughput and runtime) of POP, NCFlow, and CSPF, as well as the optimal baseline (no partitioning). Further traffic engineering experiments will append to these CSV files.

## Figure 11: Max-Flow Traffic Engineering, (Network x Traffic Matrix)

To reproduce Figure 11, run the following command from `traffic_engineering/benchmarks`:
```bash
./pop.py --slices 0 --tm-models uniform gravity bimodal poisson-high-inter --split-fractions 0 --num-subproblems 16 --split-methods random --obj total_flow

./pop.py --slices 0 --tm-models poisson-high-intra --split-fractions 0.75 --num-subproblems 16 --split-methods random --obj total_flow

./path_form.py --slices 0 --obj total_flow
```

This script runs over 300 experiments, and will take a long time to complete.

## Figure 12: Max Concurrent Flow Traffic Engineering

To reproduce Figure 12, run the following command from `traffic_engineering/benchmarks`:
```bash
./pop.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --split-fractions 0 --num-subproblems 4 16 64 --split-methods random --obj mcf

./path_form.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --obj mcf
```

## Figure 13: Load balancing

To reproduce Figure 13 (that is, show the impact of POP on a MILP optimization problem
for load balancing), run the following command from `load_balancing`:

```bash
java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards 1024 -numServers 128 -benchmark base

java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards 1024 -numServers 128 -numSplits 4 -benchmark split

java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards 1024 -numServers 128 -numSplits 16 -benchmark split

java -jar target/POP-1.0-SNAPSHOT-fat-tests.jar -numShards 1024 -numServers 128 -benchmark heuristic
```

## Figure 14: Client Splitting, Traffic Engineering

To reproduce Figure 14, run the following command from `traffic_engineering/benchmarks`:
```bash
./pop.py --slices 0 --tm-models poisson-high-intra --split-fractions 0 0.5 1.0 --num-subproblems 16 --split-methods random --obj total_flow

./pop.py --slices 0 --tm-models gravity --split-fractions 0 1.0 --num-subproblems 16 --split-methods random --obj total_flow

./path_form.py --slices 0 --tm-models poisson-high-intra gravity --obj total_flow
```

This script runs many experiments, and will take a long time to complete.

## Figure 15: Different Client Partition Methods, Traffic Engineering

To reproduce Figure 15, run the following command from `traffic_engineering/benchmarks`:
```bash
./pop.py --slices 0 --topos Cogentco.graphml --tm-models gravity --scale-factors 64 --split-fractions 0 --num-subproblems 4 8 16 --split-methods random means skewed --obj total_flow

./path_form.py --slices 0 --topos Cogentco.graphml --tm-models gravity --scale-factors 64 --obj total_flow
```
