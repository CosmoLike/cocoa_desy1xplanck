"""Accuracy advisory checks A1-A6: default vs high-accuracy settings.

Every reference of these tests is computed with the examples' default
numerical settings. These checks answer: how much numerical error do
those defaults carry? Each one re-evaluates a frozen configuration at
its frozen point with the numerical settings pushed far beyond the
defaults (cosmolike: accuracyboost 3, internal_accuracyboost 2,
integration_accuracy 10, lmax 200000, kmax_boltzmann 40; CAMB:
AccuracyBoost 2.0, k_per_logint 50, kmax 50; the exact values are
HIGH_ACCURACY_LIKELIHOOD and HIGH_ACCURACY_CAMB_EXTRA_ARGS of
cocoa_test_utils) and reports

    delta chi2 = chi2(high accuracy) - chi2(default, frozen)

There is no pass/fail: how much numerical error an analysis tolerates
is a judgment call. The six checks cover the three probes with both
IA models:

  A1. cosmic shear (example1), NLA      A2. cosmic shear, TATT
  A3. 2x2pt (example2_2x2pt), NLA       A4. 2x2pt, TATT
  A5. 6x2pt (example2), NLA             A6. 6x2pt, TATT

The checks evaluate against data vectors generated at the frozen
point when the frozen state was built (the synthetic NLA vector, or the
TATT-generated one for the TATT checks), so the chi2 sits at a minimum
and the delta is a stable, quadratic response instead of a linear one.

A high-accuracy evaluation takes minutes, not seconds: the whole file
is far slower than the other tests. To run only this file (from the
Cocoa/ folder, cocoa environment active, start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_accuracy.py

and to run the other tests without it:

    python -m pytest ./projects/desy1xplanck/tests --ignore \\
        ./projects/desy1xplanck/tests/data_vector/test_accuracy.py
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
# the advisory class; assigning its return value to the usual
# module-level name is what makes pytest collect it here, with
# unchanged test ids. Each roster row is (method name, advisory
# label, example, TATT flag, report label); knob_scan binds the
# example the one-knob-at-a-time scan evaluates.
TestAccuracyAdvisory = u.accuracy_suite(
    u, name="TestAccuracyAdvisory",
    checks=(
        ("test_a1_cosmic_shear_nla", "A1", "example1",
         False, "example1 (cosmic shear, NLA)"),
        ("test_a2_cosmic_shear_tatt", "A2", "example1",
         True, "example1 (cosmic shear, TATT)"),
        ("test_a3_2x2pt_nla", "A3", "example2_2x2pt",
         False, "example2_2x2pt (2x2pt, NLA)"),
        ("test_a4_2x2pt_tatt", "A4", "example2_2x2pt",
         True, "example2_2x2pt (2x2pt, TATT)"),
        ("test_a5_6x2pt_nla", "A5", "example2",
         False, "example2 (6x2pt, NLA)"),
        ("test_a6_6x2pt_tatt", "A6", "example2",
         True, "example2 (6x2pt, TATT)"),
    ),
    knob_scan="example2")

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
