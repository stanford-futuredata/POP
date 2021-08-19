#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("../..")

from lib.plot_utils import print_stats
from lib.cdf_utils import plot_cdfs, get_ratio_dataframes


def plot_mcf_cdfs(title=""):
    random_32 = 'split_method == "random" and num_subproblems == 32'
    means_32 = 'split_method == "means" and num_subproblems == 32'
    random_16 = 'split_method == "random" and num_subproblems == 16'
    means_16 = 'split_method == "means" and num_subproblems == 16'
    random_4 = 'split_method == "random" and num_subproblems == 4'
    means_4 = 'split_method == "means" and num_subproblems == 4'

    ratio_dfs = get_ratio_dataframes(
        "path-form.csv",
        "pop.csv",
        pop_parent_query_str=None,
        pop_query_strs=[random_32, means_32, random_16, means_16, random_4, means_4],
    )

    pop_random_32_df = ratio_dfs["POP"][random_32]
    pop_random_16_df = ratio_dfs["POP"][random_16]
    pop_random_4_df = ratio_dfs["POP"][random_4]

    pop_means_32_df = ratio_dfs["POP"][means_32]
    pop_means_16_df = ratio_dfs["POP"][means_16]
    pop_means_4_df = ratio_dfs["POP"][means_4]

    print_stats(pop_random_32_df, "Random, 32", ["obj_val_ratio", "speedup_ratio"])
    print_stats(pop_means_32_df, "Power-of-two, 32", ["obj_val_ratio", "speedup_ratio"])

    print_stats(pop_random_16_df, "Random, 16", ["obj_val_ratio", "speedup_ratio"])
    print_stats(pop_means_16_df, "Power-of-two, 16", ["obj_val_ratio", "speedup_ratio"])

    print_stats(pop_random_4_df, "Random, 4", ["obj_val_ratio", "speedup_ratio"])
    print_stats(pop_means_4_df, "Power-of-two, 4", ["obj_val_ratio", "speedup_ratio"])

    # Plot CDFs
    plot_cdfs(
        [
            pop_random_32_df["speedup_ratio"],
            pop_means_32_df["speedup_ratio"],
            pop_random_16_df["speedup_ratio"],
            pop_means_16_df["speedup_ratio"],
            pop_random_4_df["speedup_ratio"],
            pop_means_4_df["speedup_ratio"],
        ],
        [
            "Random, 32",
            "Power-of-two, 32",
            "Random, 16",
            "Power-of-two, 16",
            "Random, 4",
            "Power-of-two, 4",
        ],
        "speedup-cdf-mcf-{}".format(title),
        x_log=True,
        x_label=r"Speedup, relative to PF4 (log scale)",
        bbta=(0, 0, 1, 1.3),
        figsize=(9, 5.5),
        ncol=3,
        title=title,
    )

    plot_cdfs(
        [
            pop_random_32_df["obj_val_ratio"],
            pop_means_32_df["obj_val_ratio"],
            pop_random_16_df["obj_val_ratio"],
            pop_means_16_df["obj_val_ratio"],
            pop_random_4_df["obj_val_ratio"],
            pop_means_4_df["obj_val_ratio"],
        ],
        [
            "Random, 32",
            "Power-of-two, 32",
            "Random, 16",
            "Power-of-two, 16",
            "Random, 4",
            "Power-of-two, 4",
        ],
        "min-frac-flow-cdf-mcf-{}".format(title),
        x_log=False,
        x_label=r"Min Frac. Flow, relative to PF4",
        bbta=(0, 0, 1, 1.3),
        figsize=(9, 5.5),
        ncol=3,
        title=title,
    )


if __name__ == "__main__":
    plot_mcf_cdfs()
