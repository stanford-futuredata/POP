#! /usr/bin/env bash

set -e
set -x

conda activate traffic_engineering
./pop.py --slices 0 \
    --topos Kdl.graphml \
    --scale-factors 16 \
    --tm-models gravity \
    --split-fractions 0 \
    --num-subproblems 4 16 64 \
    --split-methods random \
    --obj mcf

./path_form.py --slices 0 \
    --topos Kdl.graphml \
    --scale-factors 16 \
    --tm-models gravity \
    --obj mcf