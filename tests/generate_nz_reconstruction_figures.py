"""Redraw this survey's n(z) figure without a cosmology calculation.

The JSON file contains per-distribution area measurements; the NPZ holds
histogram cells and sampled C-reader curves. The shared core plotter selects
one close-up and draws all distributions' probability changes. See
studies/README.md for the selection rule and the meaning of both panels.
"""
from pathlib import Path
import sys

# Resolve paths from this script, not the terminal's working directory.
# The shared plotting implementation belongs to core; the project supplies
# only the locations of its own saved measurements and output figure.
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
