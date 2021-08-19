#! /usr/bin/env bash

set -e
set -x

julia ../ext/teavar/run_teavar_star.jl b4-teavar.json 2 0.95 x EDInvCap4 topo_n2.txt 2 1 > teavar_star.txt
