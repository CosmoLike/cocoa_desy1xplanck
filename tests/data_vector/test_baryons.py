"""Baryonic feedback drift tests BD1-BD7: frozen-vector pinning.

Each test evaluates the example1 configuration (NLA) with the bfmt
theory block computing one feedback method (the block multiplies the
nonlinear matter power spectrum by a baryonic suppression factor
S(k, z)), against that method's frozen data vector: the
default-settings theory prediction written at freeze time by
generate_frozen_reference.py --baryons, at the frozen fiducial point
plus the method's cosmology override
(cocoa_test_utils.BARYON_POINT_OVERRIDES). At freeze time the chi2
against that vector was zero by construction, so the assertion

    chi2 <= CHI2_TOLERANCE (0.2)

pins the whole feedback pipeline: a failure means cosmolike or the
bfmt theory block changed its prediction since the freeze. This is
the reference-test idea (test_example1.py) applied to the feedback
pipeline, and it complements test_accuracy_baryons.py: the accuracy
checks regenerate their vector on the fly per run, so they measure
numerical settings and can never see drift; these tests hold the
frozen vector still, so they measure drift and nothing else. The
seven tests cover every method the bfmt theory block implements:

  BD1. SP(k), power-law fb relation     BD2. SP(k), Akino et al. 2022
  BD3. SP(k), double power-law relation BD4. BCEmu
  BD5. FlamingoBaryonResponseEmulator   BD6. BACCOemu
  BD7. BCemu2025

To run only this file (from the Cocoa/ folder, cocoa environment
active, start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_baryons.py
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


class TestBaryonDrift(unittest.TestCase):
    """Drift tests BD1-BD7: the feedback pipeline against its freeze.

    setUpClass runs once: it moves to ROOTDIR and verifies every
    frozen file against the SHA-256 manifest before any physics runs.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def _baryon_drift_check(self, name, baryon, label):
        """One method's chi2 against its frozen feedback vector.

        Arguments:
          name   = the test label (BD1-BD7) for the report.
          baryon = a label of cocoa_test_utils.BARYON_METHODS.
          label  = one line naming the feedback method.

        Returns:
          nothing; prints the chi2 between two lines of 66 dashes.

        Raises:
          AssertionError when the chi2 exceeds CHI2_TOLERANCE.
        """
        chi2 = u.baryon_drift_chi2(baryon)
        # the triple-quoted f-string prints a framed report; {'-' * 66}
        # inserts a line of 66 dashes
        print(f"""
{'-' * 66}
DRIFT: {name}: {label}
  chi2 against the frozen feedback vector = {chi2:.6f}
  (zero at freeze time; tolerance {u.CHI2_TOLERANCE})
{'-' * 66}""", flush=True)
        self.assertLessEqual(
            chi2, u.CHI2_TOLERANCE,
            f"{name}: chi2 {chi2:.6f} exceeds the tolerance "
            f"{u.CHI2_TOLERANCE}; cosmolike or the bfmt theory block "
            "changed its prediction since the freeze")

    def test_bd1_spk_power_law(self):
        """BD1: SP(k) with the power-law fb relation."""
        self._baryon_drift_check("BD1", "spk power law",
                                 "SP(k), power-law fb relation")

    def test_bd2_spk_akino(self):
        """BD2: SP(k) with the Akino et al. 2022 fb relation."""
        self._baryon_drift_check("BD2", "spk akino",
                                 "SP(k), Akino et al. 2022")

    def test_bd3_spk_double_power_law(self):
        """BD3: SP(k) with the double power-law fb relation."""
        self._baryon_drift_check("BD3", "spk double power law",
                                 "SP(k), double power-law fb relation")

    def test_bd4_bcemu(self):
        """BD4: BCEmu."""
        self._baryon_drift_check("BD4", "bcemu", "BCEmu")

    def test_bd5_flamingo(self):
        """BD5: FlamingoBaryonResponseEmulator."""
        self._baryon_drift_check("BD5", "flamingo",
                                 "FlamingoBaryonResponseEmulator")

    def test_bd6_baccoemu(self):
        """BD6: BACCOemu."""
        self._baryon_drift_check("BD6", "baccoemu", "BACCOemu")

    def test_bd7_bcemu2025(self):
        """BD7: BCemu2025."""
        self._baryon_drift_check("BD7", "bcemu2025", "BCemu2025")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
