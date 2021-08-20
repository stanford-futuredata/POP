# Setup
1. Install dependencies:
  ```bash
  sudo apt-get install -y build-essential cmake python-dev python3-dev unzip zip
  ```
2. Install [Miniconda with Python 3.8](https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh)
3. Run `conda env create -f environment.yml` in `traffic_engineering/`

# Traffic Engineering
1. cd `traffic_engineering/`
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
   ssh -D 1337 -f -C -q -N [your_university_username]@[domain_of_machine_in_university_network]
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

# Load Balancing
1. cd `load_balancing`
2. `mvn package`
3. Run the experiment show in Figure 13: 
```./figure13.sh```