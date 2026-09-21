"""Accuracy advisory checks A1-A6: default vs high-accuracy settings.

Every reference in this suite is computed with the examples' default
numerical settings. These checks answer: how much numerical error do
those defaults carry? Each one re-evaluates a frozen configuration at
its frozen point with the numerical knobs pushed far beyond the
defaults (cosmolike: accuracyboost 2, integration_accuracy 10,
lmax 200000, kmax_boltzmann 40; CAMB: AccuracyBoost 2.0,
k_per_logint 50, kmax 50; the exact values live in
cocoa_test_utils.HIGH_ACCURACY_*) and reports

    delta chi2 = chi2(high accuracy) - chi2(default, frozen)

There is NO pass/fail: how much numerical error an analysis tolerates
is a judgment call. The six checks cover the three probes with both
IA models:

  A1. cosmic shear (example1), NLA      A2. cosmic shear, TATT
  A3. 2x2pt (example2_2x2pt), NLA       A4. 2x2pt, TATT
  A5. 6x2pt (example2), NLA             A6. 6x2pt, TATT

The TATT checks evaluate against their configuration's
TATT-generated data vector (written at freeze time), so the
chi2 sits at a minimum and the delta is a stable, quadratic response
instead of a linear one.

A high-accuracy evaluation takes minutes, not seconds: the whole file
is far slower than the rest of the suite. To run only this file (from
the Cocoa/ folder, cocoa environment active, start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/test_accuracy.py

and to run the rest of the suite without it:

    python -m pytest ./projects/desy1xplanck/tests --ignore \\
        ./projects/desy1xplanck/tests/test_accuracy.py
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


class TestAccuracyAdvisory(unittest.TestCase):
    """Advisory checks A1-A6, sharing one frozen-state verification.

    setUpClass runs once: it moves to ROOTDIR, verifies every frozen
    file against the SHA-256 manifest (an edited frozen state must
    fail loudly before any physics runs), and loads the frozen
    reference chi2 values.
    """

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def _accuracy_check(self, name, example, tatt, label):
        """Evaluate one configuration at high accuracy and report.

        Arguments:
          name    = the advisory label (A1-A6) for the report.
          example = a key of cocoa_test_utils.EXAMPLES (exact-physics
                    configurations only).
          tatt    = True evaluates the TATT variant against the
                    TATT-generated data vector, False the NLA one.
          label   = one line naming the probe and IA model.
        """
        chi2_high = u.single_model_chi2(example, tatt, high_accuracy=True)
        suffix = "tatt" if tatt else "nla"
        default_ref = self.reference[f"{example}_{suffix}"]
        u.report_accuracy(f"{name}: {label}", chi2_high, default_ref)

    def test_a0_one_knob_at_a_time(self):
        """K scan: each accuracy knob alone on example2, NLA.

        Advisory: each knob's chi2 and its difference to the frozen
        default reference print as the scan runs. A knob whose delta
        rivals the all-knobs delta is the driver; a knob whose delta
        explodes (orders of magnitude beyond the others) signals an
        interface breakdown, not a numerics improvement.
        """
        default_ref = self.reference["example2_nla"]
        print("", flush=True)
        for label, _, _ in u.ACCURACY_KNOBS:
            chi2 = u.single_model_chi2("example2", False, knob=label)
            u.report_knob(label, chi2, default_ref)

    def test_a1_cosmic_shear_nla(self):
        """A1: cosmic shear, NLA, default vs high accuracy."""
        self._accuracy_check("A1", "example1", False,
                             "example1 (cosmic shear, NLA)")

    def test_a2_cosmic_shear_tatt(self):
        """A2: cosmic shear, TATT, default vs high accuracy."""
        self._accuracy_check("A2", "example1", True,
                             "example1 (cosmic shear, TATT)")

    def test_a3_2x2pt_nla(self):
        """A3: 2x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A3", "example2_2x2pt", False,
                             "example2_2x2pt (2x2pt, NLA)")

    def test_a4_2x2pt_tatt(self):
        """A4: 2x2pt, TATT, default vs high accuracy."""
        self._accuracy_check("A4", "example2_2x2pt", True,
                             "example2_2x2pt (2x2pt, TATT)")

    def test_a5_6x2pt_nla(self):
        """A5: 6x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A5", "example2", False,
                             "example2 (6x2pt, NLA)")

    def test_a6_6x2pt_tatt(self):
        """A6: 6x2pt, TATT, default vs high accuracy."""
        self._accuracy_check("A6", "example2", True,
                             "example2 (6x2pt, TATT)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
