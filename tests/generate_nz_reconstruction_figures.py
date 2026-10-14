"""Regenerate the n(z) reconstruction and bin-area figures from saved test results.

Run from an active Cocoa environment. This reads the committed probability
audit; it neither changes frozen references nor reruns a cosmology.
"""
from pathlib import Path
import sys

TESTS = Path(__file__).resolve().parent
CORE = TESTS.parents[2] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(CORE))
from cocoa_nz_figures import plot_nz_reconstruction


if __name__ == "__main__":
    plot_nz_reconstruction(
        report_file=TESTS / "nz_reconstruction.json",
        arrays_file=TESTS / "nz_probability_data.npz",
        output_directory=TESTS)
