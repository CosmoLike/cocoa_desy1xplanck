"""Advisory checks NL1-NL2: Halofit vs EuclidEmulator2, per probe.

The likelihoods can source the nonlinear matter power from CAMB's
Takahashi halofit (non_linear_emul: 2, the setting of the frozen
configurations) or from EuclidEmulator2 (EE2, non_linear_emul: 1, an
emulator of the nonlinear boost P_nl/P_lin). Both checks
evaluate their data vector with both sources at ten fixed
cosmologies across the omegam/ns/As space
(NONLINEAR_COMPARISON_POINTS; every other parameter stays at the
frozen fiducial) and report, at each cosmology, the chi2 of the
Halofit vector against the EE2 vector (delta^T C^-1 delta; the EE2
vector is that cosmology's fiducial, so the baseline is zero by
construction and no stored data vector enters the metric).

NL1 runs the sweep on cosmic shear (example1); NL2 runs it on the
6x2pt likelihood (example2, desy1xplanck.combo_6x2pt), where every
cosmology also evaluates the CMB parts of the vector, so its sweep
costs more per point.

There is no pass/fail: the numbers say how much of the statistical
error budget the Halofit-vs-emulator difference consumes under the
chosen scale cuts, which answers whether Halofit can be used on real
data with this mask. The checks read the --mask option of the
comparison sweeps (conftest.py): --mask=frozen (the default) keeps
the shipped 6x2pt scale-cut mask of the frozen configurations, and
--mask=ones keeps every data point (no scale cuts). NL2 aborts
under ones: with every point unmasked the shipped 6x2pt covariance
is not positive definite and cosmolike refuses it (see
tests/data_vector/README.md).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_nonlinear.py

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_nonlinear.py \\
        --mask=ones
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factories (cosmolike_core/cocoa_testing_suites.py) build
# the classes; assigning each return value to the usual module-level
# name is what makes pytest collect it here, with unchanged test ids.
TestHalofitVsEE2 = u.nonlinear_suite(
    u, name="TestHalofitVsEE2",
    cases=(
        ('test_nl1_halofit_vs_ee2_cosmic_shear', 'NL1', 'example1', 'example1 (cosmic shear'),
        ('test_nl2_halofit_vs_ee2_6x2pt', 'NL2', 'example2', 'example2 (6x2pt'),
    ))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
