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

### Software Dependencies for Ubuntu 16.04

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
3. Download and install CPLEX 12.1 free academic version (requires an IBM account,
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


### Software Dependencies for Ubuntu 18.04

1. Install `apt` dependencies:
  ```bash
  sudo apt update && sudo apt -y upgrade
  sudo apt install -y openjdk-11-jre-headless default-jre build-essential cmake python-dev python3-dev maven unzip zip htop g++ gcc libnuma-dev make numactl zlib1g-dev
  ```
2. Install [Miniconda with Python 3.8](https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh
```
3. Download and install CPLEX 12.1 free academic version (requires an IBM account,
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

## Figure 2: Max-Min Fairness Policy with Space Sharing with No Trace

To reproduce Figure 2 in the paper, run the following command from `cluster_scheduling`:

```bash
python figure2.py
```

## Figure 6: Max-Min Fairness Policy with Space Sharing with Trace

To reproduce Figure 6 in the paper (that is, evaluate the max-min fairness policy presented
in Section 4.1 of the paper), run the following command from `cluster_scheduling/scheduling`
(fill in the output directory as appropriate, this needs to be created beforehand):

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 1000 -e 1500 -l /path/to/log/directory -j 1 -p max_min_fairness_packed --seeds 1 -c 32:32:32 -a 6.4 -b 6.4 -n 1 --num_sub_problems 1 2 4 8 --solver MOSEK
```

Partial output of this script looks like this:

```
[2021-09-28 17:08:09.342446] Running 4 total experiment(s)...
[2021-09-28 17:08:09.360697] [Experiment ID:  0] Configuration: cluster_spec=v100:32|p100:32|k80:32, policy=MaxMinFairness_Packing, seed=1, lam=562.500000, profiling_percentage=1.000000, num_reference_models=26, num_sub_problems=1
...
```

This can take about a day to complete. We suggest using `tmux`. This creates a
collection of logfiles under `/path/to/log/directory` (one for each independent
case run).

The generated output logfiles can be analyzed using the postprocessing script
available at `cluster_scheduling/process_logs.py` (the output directory used
above needs to be provided to this script as a command line argument):

```bash
> python process_logs.py -l /path/to/log/directory
V100s	P100s	K80s	Policy			K	Seed	Lambda	Metric	Runtime
32	32	32	max_min_fairness_packed	2	1	562.5	28.765	0.443
32	32	32	max_min_fairness_packed	4	1	562.5	27.86	0.185
32	32	32	max_min_fairness_packed	1	1	562.5	28.871	1.269
32	32	32	max_min_fairness_packed	8	1	562.5	28.809	0.117
```

We have also provided a notebook with code for analyzing this post-processed data. Pipe the `stdout` to a file (e.g., `max_min_fairness_packed.tsv`) and then point `cluster_scheduling/figures.ipynb` at this file.

## Figure 8: Minimize Makespan

To reproduce Figure 9 in the paper, run the following command from `cluster_scheduling/scheduler`:

```bash
python -u scripts/sweeps/run_sweep_static.py -l /path/to/log/directory -j 1 -p min_total_duration_perf --seeds 1 -c 32:32:32 -a 700 -b 700 -n 1 --generate-multi-gpu-jobs --num_sub_problems 1 2 4 8 --solver MOSEK
```

## Figure 9: Max-Flow Traffic Engineering, Single Network

To reproduce Figure 10, run the following command from `traffic_engineering/benchmarks`:
```bash
./pop.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --split-fractions 0 --num-subproblems 4 16 64 --split-methods random --obj total_flow

./ncflow.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --obj total_flow

./cspf.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --obj total_flow

./path_form.py --slices 0 --topos Kdl.graphml --scale-factors 16 --tm-models gravity --obj total_flow
```

This will create CSV files containing the results (including throughput and runtime) of POP, NCFlow, and CSPF, as well as the optimal baseline (no partitioning). Further traffic engineering experiments will append to these CSV files.

## Figure 10: Max-Flow Traffic Engineering, (Network x Traffic Matrix)

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
