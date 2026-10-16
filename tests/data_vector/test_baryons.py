"""Baryonic feedback drift tests BD1-BD7: frozen-vector pinning.

The class these tests live in is built by the shared factory
baryon_drift_suite (cosmolike_core/cocoa_testing_suites.py), and what
each test means physically is documented in the cosmolike_core
README's shared data-vector checks, baryonic-feedback section. In
short: each test evaluates this project's example1 configuration
(cosmic shear, NLA) with the bfmt theory block computing one feedback
method, against that method's frozen data vector (written at freeze
time by generate_frozen_reference.py --baryons, when the chi2 against
it was zero by construction), so the assertion

    chi2 <= chi2_tolerance

pins the whole feedback pipeline against drift. The methods and their
fixed parameter points live in cocoa_test_utils.BARYON_METHODS and
BARYON_POINT_OVERRIDES.

One evaluation per test: expect minutes per test. To run only this
file (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_baryons.py
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
TestBaryonDrift = u.baryon_drift_suite(u)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
