"""Baryonic feedback accuracy checks B1-B7: default vs high accuracy.

Each check evaluates the example1 configuration (NLA) with the bfmt
theory block switched on for one of its feedback methods, at the
default numerical settings and again with every accuracy knob pushed
(cocoa_test_utils.HIGH_ACCURACY_*), and reports

    delta chi2 = chi2(high accuracy) - chi2(default)

Both values are computed in this run: switching the feedback on
changes the data-vector prediction, so the frozen no-feedback
reference does not apply here. Advisory like test_accuracy.py, with
NO pass/fail: the question answered is whether the numerical error
of the default settings stays harmless when the nonlinear power
spectrum carries a baryonic suppression.

The seven checks cover every method the bfmt theory block
implements:

  B1. SP(k), power-law fb relation      B2. SP(k), Akino et al. 2022
  B3. SP(k), double power-law relation  B4. BCEmu
  B5. FlamingoBaryonResponseEmulator    B6. BACCOemu
  B7. BCemu2025

Each parameter point is fixed (the SP(k) points are pyspk's
documented examples; the emulator points are the fiducial values
quoted in the example yamls); the exact values live in
cocoa_test_utils.BARYON_METHODS.

Two readings to keep in mind. First, the frozen data vector carries
no feedback, so switching a method on moves the chi2 far from its
minimum, and the delta rides a steep slope: expect larger values
than the at-minimum checks of test_accuracy.py report, and compare
the methods against each other rather than against the band. Second,
a method may REJECT the frozen fiducial when it violates the
method's own training box (BACCOemu's omega_baryon boundary sits at
omegab = 0.04001, for example); the check then reports the rejection
as documented behavior instead of a number.

This file adds to test_accuracy.py and does not replace or modify
it. Two evaluations per check, one of them at high accuracy: expect
minutes per check. To run only this file (from the Cocoa/ folder,
cocoa environment active, start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/test_accuracy_baryons.py
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


class TestBaryonAccuracyAdvisory(unittest.TestCase):
    """Advisory checks B1-B7: accuracy with baryonic feedback on.

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
          name   = the advisory label (B1-B7) for the report.
          baryon = a label of cocoa_test_utils.BARYON_METHODS.
          label  = one line naming the feedback method.
        """
        chi2_default = u.single_model_chi2("example1", False,
                                           baryon=baryon)
        if chi2_default is None:
            # the method rejected the frozen fiducial: a training-box
            # violation, reported by the warning above naming the
            # parameter; the rejection path working IS the result
            print(f"""
{'-' * 66}
ACCURACY: {name}: {label}
  the method rejected the frozen fiducial (training-box violation;
  the warning above names the parameter). The rejection path works
  as documented; no delta is measured for this method.
{'-' * 66}""", flush=True)
            return
        chi2_high = u.single_model_chi2("example1", False,
                                        high_accuracy=True,
                                        baryon=baryon)
        u.report_accuracy(f"{name}: {label}", chi2_high, chi2_default,
                          default_name="default, this run")

    def test_b1_spk_power_law(self):
        """B1: SP(k) with the power-law fb relation."""
        self._baryon_accuracy_check("B1", "spk power law",
                                    "SP(k), power-law fb relation")

    def test_b2_spk_akino(self):
        """B2: SP(k) with the Akino et al. 2022 fb relation."""
        self._baryon_accuracy_check("B2", "spk akino",
                                    "SP(k), Akino et al. 2022")

    def test_b3_spk_double_power_law(self):
        """B3: SP(k) with the double power-law fb relation."""
        self._baryon_accuracy_check("B3", "spk double power law",
                                    "SP(k), double power-law fb relation")

    def test_b4_bcemu(self):
        """B4: BCEmu."""
        self._baryon_accuracy_check("B4", "bcemu", "BCEmu")

    def test_b5_flamingo(self):
        """B5: FlamingoBaryonResponseEmulator."""
        self._baryon_accuracy_check("B5", "flamingo",
                                    "FlamingoBaryonResponseEmulator")

    def test_b6_baccoemu(self):
        """B6: BACCOemu."""
        self._baryon_accuracy_check("B6", "baccoemu", "BACCOemu")

    def test_b7_bcemu2025(self):
        """B7: BCemu2025."""
        self._baryon_accuracy_check("B7", "bcemu2025", "BCemu2025")


if __name__ == "__main__":
    unittest.main(verbosity=2)
