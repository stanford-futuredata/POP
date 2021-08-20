# Setup
0. Mount the EBS volume (TODO: specifics)
1. Install dependencies:
  ```bash
  sudo apt-get install -y build-essential cmake python-dev python3-dev openjdk-11-jdk maven unzip zip htop
  ```
2. Install [Miniconda with Python 3.8](https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh)
3. Run `conda env create -f environment.yml` in `traffic_engineering/`

## Traffic Engineering
1. `cd traffic_engineering/`
2. `conda activate traffic_engineering`
3. `pip install -r requirements.txt`
4. Download and install [gurobi 8.1.1](https://packages.gurobi.com/8.1/gurobi8.1.1_linux64.tar.gz). Unzip the tarfile in your home directory and
add/modify the following environment variables in your .bashrc:
```bash
export GUROBI_HOME=$HOME/guorbi811/linux64
export PATH=$GUROBI_HOME/bin:$PATH
```
Source your .bashrc so that these variables are now available.
5. Obtain a [free Gurobi academic
   license](https://www.gurobi.com/academia/academic-program-and-licenses/).
   Note that you will have to use `grbgetkey`, which should be available to you
   once you've completed the previous step. This will require creating a Gurobi
   account and accepting their EULA. After doing so, Gurobi will give you a command
   to run to download the key to your machine; for example:
   ```bash
   grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   ```

   This will NOT work, because Gurobi requires that the command be run on a
   machine that is connected to a university network. To get around this, you
   will have to set up SOCKS proxy via `ssh`:
   ```bash
   ssh -D 1337 -f -C -q -N [your_university_username]@[domain_or_public_ip_of_machine_in_university_network]
   ```

  * `-D`: Tells `ssh` that we want a SOCKS tunnel on the specified port number (you can choose a number between 1025 and 65536)
  * `-f`: Forks the process to the background
  * `-C`: Compresses the data before sending it
  * `-q`: Uses quiet mode
  * `-N`: Tells `ssh` that no command will be sent once the tunnel is up

  Then, run `grbgetkey` while simultaneously setting `HTTPS_PROXY`:
  ```bash
  HTTPS_PROXY=socks5://localhost:1337 grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  ```

  That should work! You can save the Gurobi license file to the `$HOME`
  directory: `/home/ubuntu/gurobi.lic`. (You can also now safely kill the
  `ssh` proxy process at this point.)

  To confirm that the Gurobi license and installation are both setup
  correctly, run `gurobi_cl --license`; you should something to the effect
  of:
  ```bash
  Academic license - for non-commercial use only - expires 2022-08-15
  Using license file /home/ubuntu/gurobi.lic
  Set parameter LogFile to value gurobi.log
  ```
6. Run `./download.sh` to download the traffic matrices used for our benchmarks.

## Load Balancing
1. `cd load_balancing`
2. `mvn package`
3. Run the experiment show in Figure 13: 

```./figure13.sh```


# Reproducing Experiments

## Figure 6: Max-Min Fairness Policy without Space Sharing

To reproduce Figure 6 in the paper (that is, evaluate the max-min fairness policy presented
in Section XX of the paper), run the following command from `cluster_scheduling/scheduling`:

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p max_min_fairness_perf --seeds 0 1 2 -c 32:32:32 -a 6.0 -b 6.0 -n 1 --num_sub_problems 1 2 4 8
```

The output of this script looks like this:

```bash
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


## Figure 7: Max-Min Fairness Policy with Space Sharing

To reproduce Figure 7 in the paper, run the following command from `cluster_scheduling/scheduling`:

```bash
python -u scripts/sweeps/run_sweep_continuous.py -s 4000 -e 5000 -l /path/to/log/directory -j 24 -p max_min_fairness_packed --seeds 0 1 2 -c 32:32:32 -a 6.4 -b 6.4 -n 1 --num_sub_problems 1 2 4 8
```
