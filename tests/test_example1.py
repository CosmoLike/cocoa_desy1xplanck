"""Unit tests 1-4: the cosmic-shear likelihood on the frozen test data.

Cosmic shear is the correlation of galaxy shape distortions produced by
weak gravitational lensing; here it is the desy1xplanck.cosmic_shear
likelihood, evaluated on the frozen copy of example1's configuration
(see cocoa_test_utils for what "frozen" means and why). The four tests:

  1. chi2 at the frozen fiducial point, within CHI2_TOLERANCE (0.2) of
     the frozen reference value.
  2. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked between
     evaluations or OpenMP threads raced.
  3. the same comparison as test 1 with the TATT intrinsic-alignment
     model (IA_model: 1) and DES_A2_1 = 0.05, DES_BTA_1 = 0.05,
     DES_A2_2 = -1.51541 replacing the NLA point's zeros.
  4. the same race check as test 2 with the TATT model.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

EXAMPLE = "example1"


class TestExample1CosmicShear(unittest.TestCase):
    """Tests 1-4, sharing one frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR,
    verifies every frozen file against the SHA-256 manifest (an edited
    frozen state must fail loudly before any physics runs), and loads
    the frozen reference chi2 values.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_1_chi2_matches_frozen_reference(self):
        """chi2 at the frozen NLA point stays within 0.2 of the reference."""
        chi2 = u.single_model_chi2(EXAMPLE, tatt=False)
        ref = self.reference[f"{EXAMPLE}_nla"]
        u.report_chi2_test(
            1, "example1 (cosmic shear, NLA) chi2 vs frozen reference",
            chi2, ref, u.CHI2_TOLERANCE)
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_2_no_race_condition_ten_in_a_row(self):
        """The fiducial as 10th of 10 cosmologies matches a fresh run."""
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=False)
        u.report_race_test(
            2, "example1 (cosmic shear, NLA) race check: 10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")

    def test_3_chi2_matches_frozen_reference_tatt(self):
        """Test 1 repeated with the TATT IA model and nonzero A2/BTA."""
        chi2 = u.single_model_chi2(EXAMPLE, tatt=True)
        ref = self.reference[f"{EXAMPLE}_tatt"]
        u.report_chi2_test(
            3, "example1 (cosmic shear, TATT) chi2 vs frozen reference",
            chi2, ref, u.CHI2_TOLERANCE)
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"TATT chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_4_no_race_condition_ten_in_a_row_tatt(self):
        """Test 2 repeated with the TATT IA model."""
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=True)
        u.report_race_test(
            4, "example1 (cosmic shear, TATT) race check: 10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"TATT 10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
