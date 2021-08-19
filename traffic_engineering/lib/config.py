import os

TL_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
TOPOLOGIES_DIR = os.path.join(TL_DIR, "topologies")
TM_DIR = os.path.join(TL_DIR, "traffic-matrices")
TEAVAR_DATA_DIR = os.path.join(TL_DIR, "ext", "teavar", "code", "data")
TEAVAR_RUNLOGS_DIR = os.path.join(TL_DIR, "ext", "teavar", "code", "runlogs")
TEAVAR_BASELINE_RESULTS_DIR = os.path.join(
    TL_DIR, "ext", "teavar", "code", "teavar_star_plots", "data"
)
