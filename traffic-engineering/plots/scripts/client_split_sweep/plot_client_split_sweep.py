#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("../..")

from lib.cdf_utils import plot_cdfs, get_ratio_dataframes


def plot_client_split_sweep_cdfs():
    # query strings for POP DF
    e0_poisson = 'split_fraction == 0 and tm_model == "poisson-high-intra"'
    e25_poisson = 'split_fraction == 0.25 and tm_model == "poisson-high-intra"'
    e50_poisson = 'split_fraction == 0.5 and tm_model == "poisson-high-intra"'
    e75_poisson = 'split_fraction == 0.75 and tm_model == "poisson-high-intra"'
    e100_poisson = 'split_fraction == 1.0 and tm_model == "poisson-high-intra"'
    e0_gravity = 'split_fraction == 0 and tm_model == "gravity"'
    e25_gravity = 'split_fraction == 0.25 and tm_model == "gravity"'
    e50_gravity = 'split_fraction == 0.5 and tm_model == "gravity"'
    e75_gravity = 'split_fraction == 0.75 and tm_model == "gravity"'
    e100_gravity = 'split_fraction == 1.00 and tm_model == "gravity"'

    ratio_dfs = get_ratio_dataframes(
        "path-form.csv",
        "pop-total_flow-slice_0-splitsweep.csv",
        pop_parent_query_str=None,
        pop_query_strs=[
            e0_poisson,
            e25_poisson,
            e50_poisson,
            e75_poisson,
            e100_poisson,
            e0_gravity,
            e25_gravity,
            e50_gravity,
            e75_gravity,
            e100_gravity,
        ],
    )

    plot_cdfs(
        [
            ratio_dfs["POP"][e0_poisson]["speedup_ratio"],
            ratio_dfs["POP"][e50_poisson]["speedup_ratio"],
            ratio_dfs["POP"][e100_poisson]["speedup_ratio"],
            ratio_dfs["POP"][e0_gravity]["speedup_ratio"],
            ratio_dfs["POP"][e100_gravity]["speedup_ratio"],
        ],
        [
            "Poisson, 0%",
            "Poisson, 50%",
            "Poisson, 100%",
            "Gravity, 0%",
            "Gravity, 100%",
        ],
        "speedup-cdf-client_split_sweep",
        x_log=True,
        x_label=r"Speedup, relative to PF4 (log scale)",
        bbta=(0, 0, 1, 1.4),
        figsize=(9, 4.5),
        ncol=3,
    )

    plot_cdfs(
        [
            ratio_dfs["POP"][e0_poisson]["obj_val_ratio"],
            ratio_dfs["POP"][e50_poisson]["obj_val_ratio"],
            ratio_dfs["POP"][e100_poisson]["obj_val_ratio"],
            ratio_dfs["POP"][e0_gravity]["obj_val_ratio"],
            ratio_dfs["POP"][e100_gravity]["obj_val_ratio"],
        ],
        [
            "Poisson, 0%",
            "Poisson, 50%",
            "Poisson, 100%",
            "Gravity, 0%",
            "Gravity, 100%",
        ],
        "total-flow-cdf-client_split_sweep",
        x_log=False,
        x_label=r"Total Flow, relative to PF4",
        bbta=(0, 0, 1, 1.4),
        figsize=(9, 4.5),
        ncol=3,
    )


if __name__ == "__main__":
    plot_client_split_sweep_cdfs()
