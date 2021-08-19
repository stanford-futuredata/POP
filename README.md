# Solving Large-Scale Granular Resource Allocation Problems Efficiently with POP

This repository contains the source code implementation of the SOSP paper
"Solving Large-Scale Granular Resource Allocation Problems Efficiently with POP".


## Directory Structure

Code in this repository is organized by allocation problem type.

- `cluster_scheduling` contains code for the cluster scheduling problem
  formulations (max-min fairness, proportional fairness, minimize makespan).

- `load_balancing` contains code for the load balancing problem formulation.

- `traffic_engineering` contains code for the traffic engineering problem
  formulations (both maximum total flow and maximum concurrent flow).


## Getting Started

For detailed instructions on how to reproduce results from the SOSP paper,
see [EXPERIMENTS.md](EXPERIMENTS.md).
