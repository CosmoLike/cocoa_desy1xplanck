"""Unit test 18: the race check with EuclidEmulator2 on.

Cocoa pins a modified EuclidEmulator2 (the EE2_GIT_COMMIT of
set_installation_options.sh): OpenMP threading, a 1,010-redshift
capacity, the get_boost2 API with a pre-built emulator, memory-leak
fixes, and a bilinear interpolation with a border fix
(external_modules/code/euclidemu2/README.md documents them). This
test pins the threading: the fiducial evaluated fresh and again as
the 10th of 10 cosmologies on one model instance, with the
nonlinear P(k) from EE2 (non_linear_emul: 1). EE2's compute is
OpenMP-threaded, so leaked state or a thread race inside it shifts
the second fiducial value; the two must agree within
RACE_TOLERANCE (1e-4).

The other half of the EE2 coverage - the modification gate that
compiles the pre-modification build (commit ff59f66) side by side
with the installed one and compares their data vectors - runs as
lsst_y1's test 18; the modifications do not depend on the project,
so one gate serves every project.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/test_ee2.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
# insert(0, ...) puts the folder FIRST in the search order, ahead of
# every other place a same-named module could hide.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


class TestEE2Race(unittest.TestCase):
    """Test 18, with the frozen-state verification.

    setUpClass runs once before the test: it moves to ROOTDIR and
    verifies every frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: the race check compares one
    build against itself, so the frozen state only supplies the
    configuration and the data files.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x18_ee2_race_ten_in_a_row(self):
        """The fiducial with EE2 as 10th of 10 matches a fresh run.

        EE2's OpenMP-threaded compute runs inside every evaluation
        of the row, so a thread race or leaked state in it moves
        the second fiducial value.
        """
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2("example1", tatt=False,
                                           ee2=True)
        u.report_race_test(
            18, "example1 (cosmic shear, NLA+EE2) race check: 10 "
            "cosmologies in a row", fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
