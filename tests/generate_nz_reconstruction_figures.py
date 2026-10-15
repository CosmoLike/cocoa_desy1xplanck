"""Redraw the four compiled reconstruction curves from the saved manual study."""
from pathlib import Path
import sys

TESTS = Path(__file__).resolve().parent
CORE = TESTS.parents[2] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(CORE))
from cosmolike_notebook_utils.plot_nz_reconstruction import plot_nz_four_reconstructions


if __name__ == "__main__":
    directory = TESTS / "studies/results"
    plot_nz_four_reconstructions(
        report_file=directory / "probability.json",
        arrays_file=directory / "probability.npz",
        output_file=directory / "reconstructions.png")
