"""Baryonic feedback accuracy checks BF0-BF7: default vs high accuracy.

The class these checks live in is built by the shared factory
baryon_accuracy_suite (cosmolike_core/cocoa_testing_suites.py), and
what each check means physically is documented in the cosmolike_core
README's shared data-vector checks, baryonic-feedback section. In
short: each check evaluates this project's example1 configuration
(cosmic shear, NLA) with the bfmt theory block computing one feedback
method and reports the advisory delta

    delta chi2 = chi2(high accuracy) - chi2(default)

against a vector the default-settings model generated on the fly, and
BF0 scans the accuracy knobs one at a time with the Akino SP(k)
method on. The methods and their fixed parameter points live in
cocoa_test_utils.BARYON_METHODS and BARYON_POINT_OVERRIDES.

This file adds to test_accuracy.py and does not replace or modify
it. Two evaluations per check, one of them at high accuracy: expect
minutes per check. To run only this file (from the Cocoa/ folder,
cocoa environment active, start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_accuracy_baryons.py
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

# The factory builds the test class bound to this project's tables;
# assigning it to the usual module-level name is what makes pytest
# collect it here, with unchanged test ids.
TestBaryonAccuracyAdvisory = u.baryon_accuracy_suite(u)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
