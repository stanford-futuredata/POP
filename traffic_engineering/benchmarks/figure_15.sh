#! /usr/bin/env bash

set -e
set -x

conda activate traffic_engineering
./pop.py --slices 0 \
    --topos Cogentco.graphml \
    --tm-models gravity \
    --scale-factors 64 \
    --split-fractions 0 \
    --num-subproblems 4 8 16 \
    --split-methods random means skewed \
    --obj total_flow

./path_form.py --slices 0 \
    --topos Cogentco.graphml \
    --tm-models gravity \
    --scale-factors 64 \
    --obj total_flow