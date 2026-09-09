"""Shared helpers for the plot scripts: load a sweep pickle and the walkable area."""

import pickle
import sys
from pathlib import Path

from utils import setup_geometry

DEFAULT_OUTPUT_DIR = "fig_results"


def load_results(argv=None):
    """Load the sweep pickle named on the command line.

    Returns (data, stem, output_dir). `data` is the dictionary written by
    utils.save_simulation_results; fallen positions are under data["results"].
    """
    argv = sys.argv if argv is None else argv
    if len(argv) == 1:
        sys.exit(f"Usage {argv[0]} pickle_file")
    save_path = Path(argv[1])
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    return data, save_path.stem, DEFAULT_OUTPUT_DIR


def walkable_area():
    """Walkable area of the Bagh as used in the simulation."""
    return setup_geometry()[0]
