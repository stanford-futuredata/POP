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
conda create --name pop
pip install -r requirements.txt
cd scheduler; make
```

#### Traffic Engineering
```bash
cd POP/traffic_engineering
conda env create -f environment.yml
conda activate traffic_engineering
pip install -r requirements.txt
./download.sh # download the traffic matrices used in our experiments.
```

#### Load Balancing

```bash
cd POP/load_balancing
mvn package
```


# Reproducing Experiments

## Gurobi Setup
Obtain a [free Gurobi academic
license](https://www.gurobi.com/academia/academic-program-and-licenses/).  Note
that you will have to use `grbgetkey`, which should be available to you once
you've completed the steps in Setup. This will require creating a Gurobi account
and accepting their EULA. After doing so, Gurobi will give you a command to run
to download the key to your machine; for example:
```bash
grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

This will NOT work if you are not on a university network machine, since Gurobi
requires that the command be run on a machine that is connected to a university
network.

To get around this, you will have to set up SOCKS proxy via `ssh`:
```bash
ssh -D 1337 -f -C -q -N [your_university_username]@[domain_or_public_ip_of_machine_in_university_network]
```
Then, run `grbgetkey` while simultaneously setting `HTTPS_PROXY`:
```bash
HTTPS_PROXY=socks5://localhost:1337 grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

That should work! You can save the Gurobi license file to the `$HOME`
directory: `/home/ubuntu/gurobi.lic`. You can also now safely kill the
`ssh` proxy process at this point.

To confirm that the Gurobi license and installation are both setup
correctly, run `gurobi_cl --license`, which should output the path of the license file.

## MOSEK Setup

Obtain a [free MOSEK license](https://www.mosek.com/products/academic-licenses/).
Put the resulting `mosek.lic` file at `$HOME/mosek/mosek.lic`.

## Figure 2: Max-Min Fairness Policy with Space Sharing for Large Problem

To reproduce Figure 2 in the paper (that is, evaluate the max-min fairness policy presented
in Section 4.1 of the paper in isolation with 2048 jobs), run the following command from
`cluster_scheduling`:

```bash
python figure2.py | tee num_jobs=2048.out
```

This will run the max-fairness policy and its POP variants (with 2, 4, and 8 sub-problems).
It will also run Gandiva, a heuristic that we compare to. The script dumps outputs of the
form:

```
[0.08643364906311035, 0.09309506416320801, 0.058812618255615234, 0.08636689186096191, 0.018480777740478516]
[{0: 1.5698842282831222, 1: 9.241579568825575, 2: 4.931977234918527, 3: 37.24441907874906, 4: 6.417652032157853, 5: 3.931049823341792, 6: 32.79323761509924, 7: 4.931977234918902, 8: 1.5698842282831154, 9: 18.94856740357798, 10: 12.233197360582315, 11: 7.798311766932931, 12: 4.931977234918848, 13: 6.703265967155044, 14: 7.798311766932706, 15: 9.241579568825344, 16: 32.79323761509924, 17: 9.241579568825548, 18: 19.658091666362743, 19: 25.860799244781344, 20: 18.948567403577982, 21: 25.8607992447814, 22: 32.79323761510021, 23: 49.65153473676654, 24: 19.65809166636262, 25: 64.70868828991907, 26: 2.5997972671366494, 27: 19.658091666362324, 28: 3.9310498233418523, 29: 8.064938648180137, 30: 6.703265967155158, 31: 22.71582244718927}, {0: 1.4647041579582472, 2: 4.601541687330533, 4: 5.987678182392197, 6: 30.596136776746935, 8: 1.4647041579451165, 10: 11.41359030776212, 12: 4.601541687358531, 14: 7.2758360616597555, 16: 30.59613677674747, 18: 18.341026422515753, 20: 17.679039954687475, 22: 30.596136776757234, 24: 18.34102642247671, 26: 2.4256144503443555, 28: 3.667675001519033, 30: 6.254156518443241, 1: 9.562110102604034, 3: 38.536186427639244, 5: 4.067392456052443, 7: 5.103035536569936, 9: 19.605770372689378, 11: 8.06878462782129, 13: 6.935758808557147, 15: 9.562110102603947, 17: 9.562110102604123, 19: 26.757743354910716, 21: 26.75774335491067, 23: 51.37362460952818, 25: 66.95301245376373, 27: 20.33990344356843, 29: 8.344659079313644, 31: 23.50368654601271}, {0: 1.4846506226865563, 4: 6.069219080766552, 8: 1.4846506226866303, 12: 4.664205902256044, 16: 31.012797400600324, 20: 17.91979476728931, 24: 18.590796853787584, 28: 3.717621725235655, 1: 9.168388136827382, 5: 3.899916640435131, 9: 18.798498272984624, 13: 6.650177327914945, 17: 9.168388136827362, 21: 25.655986708712636, 25: 64.19620677130798, 29: 8.001065836166479, 2: 4.518923835785035, 6: 30.046802133608416, 10: 11.208666409399097, 14: 7.145202930155343, 18: 18.011724242423327, 22: 30.04680212427316, 26: 2.3820640597392075, 30: 6.141867033239036, 3: 38.97512050799075, 7: 5.161159901298792, 11: 8.160689471574116, 15: 9.67102401016267, 19: 27.062518175879294, 23: 51.958779031682916, 27: 20.57157836784196, 31: 23.77139757606815}, {0: 1.1789473246125695, 8: 1.1789473246125608, 16: 24.62697558913201, 24: 14.762779488774754, 1: 8.929825798829546, 9: 18.309359577592918, 17: 8.92982579882823, 25: 62.52581501879615, 2: 4.460791117991594, 10: 11.064474671121282, 18: 17.78001661024772, 26: 2.351420445164563, 3: 37.71497996545231, 11: 7.896838839335188, 19: 26.187534939467398, 27: 19.906459722862778, 4: 6.34975893010398, 12: 4.879801218650316, 20: 18.748108238080682, 28: 3.889462825955972, 5: 3.7937298069908136, 13: 6.469106492712816, 21: 24.95742629099698, 29: 7.783213226369135, 6: 27.25118320098214, 14: 6.4803978943980125, 22: 27.251183200981725, 30: 5.570414267614176, 7: 4.562504438418189, 15: 8.549258580559219, 23: 45.931952905181184, 31: 21.014095291971774}, {0: 0.0, 1: 5.43367116841246, 2: 2.5786536394475457, 3: 10.870500201699938, 4: 3.909662257146511, 5: 0.0, 6: 13.05771282787416, 7: 2.493791802549553, 8: 0.0, 9: 14.385277028360374, 10: 8.26742556582763, 11: 4.4609683033887535, 12: 3.033003223038409, 13: 4.384722662091917, 14: 2.345213014744705, 15: 7.159066662450499, 16: 19.750183041550553, 17: 6.039062581368526, 18: 13.338027878527582, 19: 13.3126752467904, 20: 16.40313485093915, 21: 20.9350650084345, 22: 18.101056471961684, 23: 30.439256714588208, 24: 8.451735133864368, 25: 43.346101858267566, 26: 0.0, 27: 9.499239979415592, 28: 2.051824837955005, 29: 3.3923062827990096, 30: 5.010541136660578, 31: 11.247211685299792}]
```

The first line is the list of runtimes in the order `[Exact sol., POP-2, POP-4, POP-8, Gandiva]`.
The second line is the list of dicts of effective throughputs for each of these jobs. For the
purposes of this figure, the allocation quality is the effective throughput ratio.

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

This is a merely driver program, and launches multiple experiments. In this case,
4 experiments are launched: one for each different value of `k`. This can take
about a day to complete. We suggest using `tmux`. This creates a collection of
logfiles under `/path/to/log/directory` (one for each experiment).

Each of these logfiles look like this in progress:

```
scheduler:INFO [228237.43631997466] [Micro-task succeeded]      Job ID: 153     Worker type: k80        Worker ID(s): [0]
scheduler:DEBUG [228237.43631997466] Current active completion events: dict_keys([])
scheduler:INFO [228237.43631997466] [Micro-task succeeded]      Job ID: 223     Worker type: k80        Worker ID(s): [1]
scheduler:DEBUG [228237.43631997466] Current active completion events: dict_keys([])
scheduler:INFO [228237.43631997466] [Micro-task succeeded]      Job ID: 74      Worker type: p100       Worker ID(s): [32]
scheduler:DEBUG [228237.43631997466] Current active completion events: dict_keys([])
scheduler:INFO [228237.43631997466] [Micro-task succeeded]      Job ID: 258     Worker type: p100       Worker ID(s): [41]
scheduler:DEBUG [228237.43631997466] Current active completion events: dict_keys([])
scheduler:INFO [228237.43631997466] [Micro-task succeeded]      Job ID: 352     Worker type: p100       Worker ID(s): [43]
scheduler:DEBUG [228237.43631997466] Current active completion events: dict_keys([])
scheduler:INFO [228237.43631997466] [Micro-task succeeded]      Job ID: 360     Worker type: p100       Worker ID(s): [42]
scheduler:DEBUG [228237.43631997466] Current active completion events: dict_keys([])
scheduler:INFO [228237.43631997466] [Micro-task succeeded]      Job ID: 367     Worker type: p100       Worker ID(s): [45]
scheduler:INFO [228237.43631997466] [Micro-task scheduled]      Job ID: 356     Worker type: v100       Worker ID(s): 77        Priority: 343.31        Deficit: -1152.49       Allocation:  [ k80 0.00] [p100 0.00] [v100 1.00]
scheduler:INFO [228237.43631997466] [Micro-task scheduled]      Job ID: 237     Worker type: v100       Worker ID(s): 64        Priority: 343.31        Deficit: -7192.96       Allocation:  [ k80 0.00] [p100 0.00] [v100 1.00]
scheduler:INFO [228237.43631997466] [Micro-task scheduled]      Job ID: 333     Worker type: v100       Worker ID(s): 69        Priority: 328.09        Deficit: -3762.07       Allocation:  [ k80 0.00] [p100 0.00] [v100 0.96]
scheduler:INFO [228237.43631997466] [Micro-task scheduled]      Job ID: 374     Worker type: v100       Worker ID(s): 65        Priority: 312.26        Deficit: 0.00   Allocation:  [ k80 0.00] [p100 0.00] [v100 0.91]
```

Once a particular experiment completes, experiment results are written
to the end of the relevant logfile:

```
Total duration: 2957739.167 seconds (821.59 hours)
Allocation computation times: [0.046056509017944336, ...]
Job completion times:
Job 1000: 8705.985
...
Average job completion time: 103935.129 seconds (28.87 hours)
Average job completion time (low priority): 103935.129 seconds (28.87 hours)
Mean allocation computation time: 1.2687 seconds
Cluster utilization: 0.925
```

The main metrics of interest here are `Mean allocation computation time` and
`Average job completion time`. The first is the runtime (x-axis of Figure 6),
and the second is the average JCT (y-axis of Figure 6).

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

We have also provided a notebook with code for analyzing this post-processed data.
Pipe the `stdout` of the above `process_logs.py` script to a file (e.g.,
`max_min_fairness_packed.tsv`) and then point `cluster_scheduling/figures.ipynb`
at this file.

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
