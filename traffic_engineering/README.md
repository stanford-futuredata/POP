# POP + Traffic Engineering

Code for traffic engineering integrated with POP, a fork of the [NCFlow repository](https://github.com/netcontract/ncflow.git).

Setup validated on Ubuntu 16.04.

Run `download.sh` to fetch the traffic matrices and pre-computed paths used in
our evaluation. (For confidentiality reasons, we only share TMs and paths for
topologies from the Internet Topology Zoo.)

## Dependencies
- Python 3.8 (Anaconda installation recommended)
  - See `environment.yml` for a list of Python library dependencies
  - Run `pip install -r requirements.txt` for additional non-`conda` dependencies
- Gurobi 9.1.2 (Requires a Gurobi license)

## Running experiments
The `benchmarks` directory contains a set of scripts `figureXX.sh`, which package all the commands described in `POP/EXPERIMENTS.md` for each figure. Running these scripts will generate the results for Figure XX.
