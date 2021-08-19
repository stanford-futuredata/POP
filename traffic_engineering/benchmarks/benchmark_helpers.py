from collections import defaultdict
from glob import iglob

import argparse
import os

import sys

sys.path.append("..")

from lib.partitioning import FMPartitioning, SpectralClustering

PROBLEM_NAMES = [
    "GtsCe.graphml",
    "UsCarrier.graphml",
    "Cogentco.graphml",
    "Colt.graphml",
    "TataNld.graphml",
    "Deltacom.graphml",
    "DialtelecomCz.graphml",
    "Kdl.graphml",
]
TM_MODELS = [
    "uniform",
    "gravity",
    "bimodal",
    "poisson-high-intra",
    "poisson-high-inter",
]
SCALE_FACTORS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

PATH_FORM_HYPERPARAMS = (4, True, "inv-cap")
NCFLOW_HYPERPARAMS = {
    "GtsCe.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "UsCarrier.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "Cogentco.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "Colt.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "TataNld.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "Deltacom.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "DialtelecomCz.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "Uninett2010.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "Interoute.graphml": (4, True, "inv-cap", SpectralClustering, 2),
    "Ion.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "Kdl.graphml": (4, True, "inv-cap", FMPartitioning, 3),
    "erdos-renyi-1260231677.json": (4, True, "inv-cap", FMPartitioning, 3),
}

PROBLEM_NAMES_AND_TM_MODELS = [
    (prob_name, tm_model) for prob_name in PROBLEM_NAMES for tm_model in TM_MODELS
]

PROBLEMS = []
GROUPED_BY_PROBLEMS = defaultdict(list)
HOLDOUT_PROBLEMS = []
GROUPED_BY_HOLDOUT_PROBLEMS = defaultdict(list)

for problem_name in PROBLEM_NAMES:
    if problem_name.endswith(".graphml"):
        topo_fname = os.path.join("..", "topologies", "topology-zoo", problem_name)
    else:
        topo_fname = os.path.join("..", "topologies", problem_name)
    for model in TM_MODELS:
        for tm_fname in iglob(
            "../traffic-matrices/{}/{}*_traffic-matrix.pkl".format(model, problem_name)
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
            GROUPED_BY_PROBLEMS[(problem_name, model, scale_factor)].append(
                (topo_fname, tm_fname)
            )
            PROBLEMS.append((problem_name, topo_fname, tm_fname))
        for tm_fname in iglob(
            "../traffic-matrices/holdout/{}/{}*_traffic-matrix.pkl".format(
                model, problem_name
            )
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
            GROUPED_BY_HOLDOUT_PROBLEMS[(problem_name, model, scale_factor)].append(
                (topo_fname, tm_fname)
            )
            HOLDOUT_PROBLEMS.append((problem_name, topo_fname, tm_fname))

GROUPED_BY_PROBLEMS = dict(GROUPED_BY_PROBLEMS)
for key, vals in GROUPED_BY_PROBLEMS.items():
    GROUPED_BY_PROBLEMS[key] = sorted(vals)

GROUPED_BY_HOLDOUT_PROBLEMS = dict(GROUPED_BY_HOLDOUT_PROBLEMS)
for key, vals in GROUPED_BY_HOLDOUT_PROBLEMS.items():
    GROUPED_BY_HOLDOUT_PROBLEMS[key] = sorted(vals)


def get_problems(args):
    problems = []
    for (
        (problem_name, tm_model, scale_factor),
        topo_and_tm_fnames,
    ) in GROUPED_BY_PROBLEMS.items():
        for slice in args.slices:
            if (
                ("all" in args.topos or problem_name in args.topos)
                and ("all" in args.tm_models or tm_model in args.tm_models)
                and ("all" in args.scale_factors or scale_factor in args.scale_factors)
            ):
                topo_fname, tm_fname = topo_and_tm_fnames[slice]
                problems.append((problem_name, topo_fname, tm_fname))
    return problems


def get_args_and_problems(output_csv_template, additional_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False)
    parser.add_argument("--obj", type=str, choices=["total_flow", "mcf"], required=True)
    parser.add_argument(
        "--tm-models", type=str, choices=TM_MODELS + ["all"], nargs="+", default="all",
    )
    parser.add_argument(
        "--topos", type=str, choices=PROBLEM_NAMES + ["all"], nargs="+", default="all",
    )
    parser.add_argument(
        "--scale-factors",
        type=lambda x: x if x == "all" else float(x),
        choices=SCALE_FACTORS + ["all"],
        nargs="+",
        default="all",
    )
    parser.add_argument(
        "--slices", type=int, choices=range(5), nargs="+", required=True
    )

    for additional_arg in additional_args:
        name_or_flags, kwargs = additional_arg[0], additional_arg[1]
        parser.add_argument(name_or_flags, **kwargs)
    args = parser.parse_args()
    slice_str = "slice_" + "_".join(str(i) for i in args.slices)
    output_csv = output_csv_template.format(args.obj, slice_str)
    return args, output_csv, get_problems(args)


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
