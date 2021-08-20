#! /usr/bin/env bash

set -e
set -x

./pop.py --slices 0 \
    --tm-models uniform gravity bimodal poisson-high-inter \
    --split-fractions 0 \
    --num-subproblems 16 \
    --split-methods random \
    --obj total_flow

./pop.py --slices 0 \
    --tm-models poisson-high-intra \
    --split-fractions 0.75 \
    --num-subproblems 16 \
    --split-methods random \
    --obj total_flow

./path_form.py --slices 0 \
    --obj total_flow