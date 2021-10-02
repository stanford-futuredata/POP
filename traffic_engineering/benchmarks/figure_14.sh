#! /usr/bin/env bash

set -e
set -x

conda activate traffic_engineering
./pop.py --slices 0 \
    --tm-models poisson-high-intra \
    --split-fractions 0 0.5 1.0 \
    --num-subproblems 16 \
    --split-methods random \
    --obj total_flow

./pop.py --slices 0 \
    --tm-models gravity \
    --split-fractions 0 1.0 \
    --num-subproblems 16 \
    --split-methods random \
    --obj total_flow

./path_form.py --slices 0 \
    --tm-models poisson-high-intra gravity \
    --obj total_flow
