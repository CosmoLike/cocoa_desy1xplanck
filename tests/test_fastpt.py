"""Unit test 15: cfastpt vs python FAST-PT, compared directly.

Cosmolike offers two implementations of the perturbation-theory
integrals that the TATT intrinsic-alignment model needs: cfastpt, a C
implementation built into the cosmolike interface (`IA_code: 0`, the
default), and the python FAST-PT package used through the fastpt
theory block (`IA_code: 1`).

15. example1 (cosmic shear): the SAME 30 hard-coded points
     across the intrinsic-alignment prior (FASTPT_COMPARISON_POINTS:
     20 drawn across the prior boxes plus a one-parameter-at-a-time
     family; cosmology fixed at the frozen fiducial) evaluated three
     times - with cfastpt, with FASTPT at the pass configuration
     (FASTPT_LOW_SETTINGS, hard-coded), and with FASTPT at the
     doubled boosts (FASTPT_HIGH_SETTINGS). Every block prints its theory vector at
     every point, and the CFASTPT vector is the fiducial of that
     point: its own chi2 against it is zero by construction, so the
     pass rule is the chi2 of the FASTPT(low) vector against it
     (delta^T C^-1 delta, a pure second-order deviation; a chi2
     difference against the shipped data would ride the slope
     instead). FASTPT(high)'s deviation is printed as the advisory
     FAST-PT grid response. Each configuration runs in its own
     subprocess, so no cache survives from one block to the next;
     inside a block the shared cosmology makes CAMB run once and the
     30 points cheap.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/test_fastpt.py

--high=1 repeats every block at the pushed camb/cosmolike settings
of the low-vs-high accuracy checks; the full comparison is one run
without the option and one with it. The design and the point values
are shared with lsst_y1's test 15; see that project's
tests/README.md for the full discussion.
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


class TestCfastptVsFastptSweep(unittest.TestCase):
    """Test 15, the direct CFASTPT-vs-FASTPT comparison.

    setUpClass runs once before the test: it moves to ROOTDIR and
    verifies every frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: this test compares the two
    implementations against each other, so the frozen state only
    supplies the configuration and the data files.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x15_cfastpt_vs_fastpt_sweep(self):
        """Cosmic shear: cfastpt and FASTPT agree at 30 IA points."""
        # the conftest copies the --high command line option into
        # this variable; .get with the "0" default keeps a run
        # outside pytest on the default settings unless the variable
        # is exported by hand
        high = os.environ.get("COCOA_FASTPT_HIGH", "0") == "1"
        setting = "high accuracy" if high else "default settings"
        (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
         dchi2_low, dchi2_high) = u.cfastpt_vs_fastpt_chi2s(
            "example1", high=high)
        largest = u.report_fastpt_comparison(
            15,
            f"example1 (cosmic shear, TATT, camb/cosmolike {setting}): "
            "CFASTPT vs FASTPT at 30 hard-coded points",
            chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
            dchi2_low, dchi2_high, u.FASTPT_COMPARISON_TOLERANCE)
        self.assertLess(
            largest, u.FASTPT_COMPARISON_TOLERANCE,
            msg="max chi2 of the FASTPT(low)-vs-CFASTPT data-vector "
                f"difference = {largest:.6f} over the comparison "
                f"points ({setting}); limit "
                f"{u.FASTPT_COMPARISON_TOLERANCE}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
