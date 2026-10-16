"""Unit tests 5-8: the 6x2pt likelihood on the frozen test data.

6x2pt combines six two-point correlations: the three of the galaxy
survey (cosmic shear, galaxy-galaxy lensing and galaxy clustering) and
the three that involve the Planck CMB lensing convergence (its
cross-correlations with the lens galaxies and with the source shear,
and its auto-spectrum bandpowers); here it is the
desy1xplanck.combo_6x2pt likelihood, evaluated on the frozen copy of
example2's configuration (see cocoa_test_utils for what "frozen"
means and why). The four tests:

  5. chi2 at the frozen fiducial point, within CHI2_TOLERANCE (0.2) of
     the frozen reference value.
  6. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked between
     evaluations or OpenMP threads raced.
  7. the same comparison as test 5 with the TATT intrinsic-alignment
     model (IA_model: 1), the amplitudes DES_A2_1 and DES_BTA_1 set to
     0.05 (zero at the NLA point) and DES_A2_2 = -1.51541 (the
     redshift index of A2, as at the NLA point).
  8. the same race check as test 6 with the TATT model.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector
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

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the chi2 + race test class for one frozen example; assigning its
# return value to the usual module-level name is what makes pytest
# collect it here, with unchanged test ids. The numbers name the
# methods and the printed reports; the label opens every report line.
TestExample2ThreeXTwo = u.example_suite(
    u, name="TestExample2ThreeXTwo", example="example2",
    label="example2 (6x2pt",
    numbers=("5", "6"), tatt_numbers=("7", "8"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
