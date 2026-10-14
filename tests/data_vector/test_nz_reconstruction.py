"""Original midpoint n(z) versus fixed 3x histogram preparation plus Akima.

Run with pytest -s to print the refined model distance. This uses the frozen
NLA fiducial and covariance at integration_accuracy=3, without fitting or
regenerating either. The shared test also checks the YAML opt-in and an exact
0 -> 1 -> 0 return to the default vector. See this folder's README for the
sampling study, probability changes and interpretation of Delta chi2.
"""

import os
from pathlib import Path
import sys

# Set threading before the native interface or CAMB is imported.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cocoa_test_utils as u
import cosmolike_desy1xplanck_interface as ci
from cocoa_nz_testing import compare_histogram_reconstructions


def test_nz_reconstruction():
    """Measure the two interpretations and verify selection/cache consistency."""
    level = int(os.environ.get("COCOA_NZ_INTEGRATION_ACCURACY", "3"))
    if level not in range(7):
        raise ValueError("COCOA_NZ_INTEGRATION_ACCURACY must be between 0 and 6")
    compare_histogram_reconstructions(
        harness=u, native=ci, project="desy1xplanck", example="example2",
        integration_accuracy=level)
