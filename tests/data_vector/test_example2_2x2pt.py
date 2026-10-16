"""Unit tests 11-14: the 2x2pt likelihood on the frozen test data.

2x2pt combines two of example2's six two-point correlations: galaxy
clustering and galaxy-galaxy lensing (cosmic shear and the three
CMB-lensing correlations are dropped). The
frozen configuration is example2's with the likelihood renamed to
desy1xplanck.combo_2x2pt: same options, same data files, same evaluation
point; only the probe selection inside cosmolike changes. The four
tests mirror tests 5-8 (see cocoa_test_utils for what "frozen" means):

 11. chi2 at the frozen fiducial point, within CHI2_TOLERANCE (0.2) of
     the frozen reference value.
 12. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4).
 13. the same comparison as test 11 with the TATT intrinsic-alignment
     model (IA_model: 1) and DES_A2_1 = 0.05, DES_BTA_1 = 0.05,
     DES_A2_2 = -1.51541.
 14. the same race check as test 12 with the TATT model.

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
TestExample2TwoXTwo = u.example_suite(
    u, name="TestExample2TwoXTwo", example="example2_2x2pt",
    label="example2 (2x2pt",
    numbers=("x11", "x12"), tatt_numbers=("x13", "x14"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
