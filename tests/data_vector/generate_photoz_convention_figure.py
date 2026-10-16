"""Regenerate the photo-z convention figures stored in tests/data_vector/.

The machinery and its documentation live in
cosmolike_core/cocoa_testing_figures.py (photoz_convention_figures):
the frozen cosmic-shear fiducial is evaluated under the four runtime
photo-z settings and the fractional data-vector differences against
the default are plotted per tomographic pair. This script keeps the
survey data: the binning, the scale-cut mask of the frozen contract,
and the panel grid.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/desy1xplanck/tests/data_vector/generate_photoz_convention_figure.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process
# (4 threads, the count the frozen references were generated with)
os.environ["OMP_NUM_THREADS"] = "4"

import sys

# Figures belong beside these data-vector checks. The shared harness
# and frozen inputs remain one folder above, in tests/. Resolve both
# from this script so it can be launched from any working directory.
FIGURES = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(FIGURES)
sys.path.insert(0, TESTS)
import cocoa_test_utils as u

# cocoa_testing_figures imports matplotlib, which the test shim must
# never pull in; the figure scripts import it directly instead (the
# shim import above already put the core folder on the module path)
import cocoa_testing_figures as figs

if __name__ == "__main__":
    figs.photoz_convention_figures(
        u, out_dir=FIGURES, kind="real",
        # 4 source bins, 30 log-spaced angular bins between the two
        # theta [arcmin] values, the scale-cut mask of the frozen contract, and
        # the (rows, columns) panel grid the source pairs fill
        ntomo=4, nlen=30, x_min=0.25, x_max=250.0,
        mask_file=os.path.join(TESTS, "frozen", "data",
                               "DESY3xPlanckR4_6x2pt_Maglim_baseline_archive.mask"),
        grid=(2, 5))
