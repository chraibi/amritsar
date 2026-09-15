"""Shared helpers for the plot scripts: load a sweep pickle and the walkable area."""

import pickle
import sys
from pathlib import Path

from utils import setup_geometry


def load_results(argv=None):
    """Load the sweep pickle named on the command line.

    Returns (data, stem, output_dir). `data` is the dictionary written by
    utils.save_simulation_results; fallen positions are under data["results"].
    output_dir is a `figures` directory next to the pickle.
    """
    argv = sys.argv if argv is None else argv
    if len(argv) == 1:
        sys.exit(f"Usage {argv[0]} pickle_file")
    save_path = Path(argv[1])
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    output_dir = save_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return data, save_path.stem, str(output_dir)


def walkable_area():
    """Walkable area of the Bagh as used in the simulation."""
    return setup_geometry()[0]


def save_figure(fig, path, dpi=150):
    """Save a figure as PDF and PNG next to each other with a tight bounding box.

    Returns the PDF path.
    """
    path = Path(path)
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    return path.with_suffix(".pdf")


def for_article():
    """True when `--for-article` is on the command line, and remove the flag.

    Article copies carry neither a title nor an insight line: the caption says it.
    Call before parsing the remaining arguments.
    """
    if "--for-article" not in sys.argv:
        return False
    sys.argv.remove("--for-article")
    return True
