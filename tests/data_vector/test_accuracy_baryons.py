"""Baryonic feedback accuracy checks BF1-BF7: default vs high accuracy.

Each check evaluates the example1 configuration (NLA) with the bfmt
theory block switched on for one of its feedback methods (the block
multiplies the nonlinear matter power spectrum by a baryonic
suppression factor S(k, z)). The default-settings model writes its own
theory vector during evaluation, that vector becomes the data of a
temporary dataset (so the default chi2 against it is zero by
construction, and nothing is stored in frozen/), and the
pushed-settings model evaluates at the same point against it. Its chi2
is therefore the reported quantity,

    delta chi2 = chi2(high accuracy) - chi2(default)

a pure numerics (curvature) response at the minimum. The delta is
advisory like test_accuracy.py: the question answered is whether the
numerical error of the default settings stays harmless when the
nonlinear power spectrum carries a baryonic suppression. BF0 also
runs the one-setting-at-a-time scan (ACCURACY_KNOBS) with
the Akino SP(k) method on, so a large delta names the setting causing
it.

The seven checks cover every method the bfmt theory block
implements (an fb relation is the baryon fraction of halos as a
function of halo mass, the input of SP(k)):

  BF1. SP(k), power-law fb relation      BF2. SP(k), Akino et al. 2022
  BF3. SP(k), double power-law relation  BF4. BCEmu
  BF5. FlamingoBaryonResponseEmulator    BF6. BACCOemu
  BF7. BCemu2025

Each parameter point is fixed (the SP(k) points are pyspk's
documented examples; the emulator points are the fiducial values
quoted in the example yamls); the exact values live in
cocoa_test_utils.BARYON_METHODS.

Every configuration here is measurable by construction: BACCOemu's
check evaluates at omegab = 0.049, inside its omega_baryon training
box (the floor, 0.04001, sits exactly above the fiducial
omegab = 0.04), and the double-power-law point keeps the baryon
fraction inside SP(k)'s calibrated band over the full redshift grid
(pyspk's documented example exits it at z >~ 1.4). The exact points
live in cocoa_test_utils.BARYON_METHODS and
BARYON_POINT_OVERRIDES.

This file adds to test_accuracy.py and does not replace or modify
it. Two evaluations per check, one of them at high accuracy: expect
minutes per check. To run only this file (from the Cocoa/ folder,
cocoa environment active, start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_accuracy_baryons.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process. "4"
# is REQUIRED_OMP_THREADS of cocoa_testing, the count of every worker
# subprocess: a race check needs more than one thread.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The harness stays in the parent tests/ folder (dirname applied twice to
# this file's absolute path). Add it explicitly so direct execution and
# worker processes resolve this project's stored inputs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u


class TestBaryonAccuracyAdvisory(unittest.TestCase):
    """Advisory checks BF1-BF7: accuracy with baryonic feedback on.

    setUpClass runs once: it moves to ROOTDIR and verifies every
    frozen file against the SHA-256 manifest before any physics runs.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def _baryon_accuracy_check(self, name, baryon, label):
        """Default vs high accuracy with one feedback method on.

        Arguments:
          name   = the advisory label (BF1-BF7) for the report.
          baryon = a label of cocoa_test_utils.BARYON_METHODS.
          label  = one line naming the feedback method.

        Returns:
          nothing; prints the delta between two lines of 66 dashes.

        Raises:
          AssertionError when the delta is NaN or infinite.
        """
        # the default chi2 is zero by construction (the default
        # model produced the very vector it is compared with), so the
        # pushed evaluation's chi2 is the delta; only that is printed
        delta = u.baryon_accuracy_delta(baryon)
        # delta == delta is False only for NaN (NaN differs from every
        # number, itself included), so the condition requires a finite
        # delta
        self.assertTrue(
            delta == delta and abs(delta) != float("inf"),
            f"{name}: non-finite delta")
        # the triple-quoted f-string prints a framed report; {'-' * 66}
        # inserts a line of 66 dashes
        print(f"""
{'-' * 66}
ACCURACY: {name}: {label}
  delta chi2 (high-default) = {delta:+.6f}
{'-' * 66}""", flush=True)

    def test_bf0_one_knob_at_a_time(self):
        """BF0: each accuracy setting alone, Akino SP(k) feedback on.

        Advisory: the same one-setting-at-a-time scan as test a0 of
        test_accuracy.py, with the bfmt block computing the Akino SP(k)
        suppression and the chi2 measured against that method's own
        generated vector. A setting whose delta rivals the all-settings
        delta of BF2 is the driver of the numerical error under
        feedback.
        """
        print("", flush=True)
        for label, _, _ in u.ACCURACY_KNOBS:
            # each setting's chi2 against the on-the-fly vector is its
            # delta (the default against that vector is zero)
            delta = u.baryon_accuracy_delta("spk akino", knob=label)
            print(f"  KNOB {label:30s} delta chi2 = {delta:+12.6f}",
                  flush=True)

    def test_bf1_spk_power_law(self):
        """BF1: SP(k) with the power-law fb relation."""
        self._baryon_accuracy_check("BF1", "spk power law",
                                    "SP(k), power-law fb relation")

    def test_bf2_spk_akino(self):
        """BF2: SP(k) with the Akino et al. 2022 fb relation."""
        self._baryon_accuracy_check("BF2", "spk akino",
                                    "SP(k), Akino et al. 2022")

    def test_bf3_spk_double_power_law(self):
        """BF3: SP(k) with the double power-law fb relation."""
        self._baryon_accuracy_check("BF3", "spk double power law",
                                    "SP(k), double power-law fb relation")

    def test_bf4_bcemu(self):
        """BF4: BCEmu."""
        self._baryon_accuracy_check("BF4", "bcemu", "BCEmu")

    def test_bf5_flamingo(self):
        """BF5: FlamingoBaryonResponseEmulator."""
        self._baryon_accuracy_check("BF5", "flamingo",
                                    "FlamingoBaryonResponseEmulator")

    def test_bf6_baccoemu(self):
        """BF6: BACCOemu."""
        self._baryon_accuracy_check("BF6", "baccoemu", "BACCOemu")

    def test_bf7_bcemu2025(self):
        """BF7: BCemu2025."""
        self._baryon_accuracy_check("BF7", "bcemu2025", "BCemu2025")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
