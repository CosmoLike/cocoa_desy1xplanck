"""Unit test: the runtime n(z) photo-z conventions.

The likelihood exposes two runtime knobs that control how the n(z)
table files are turned into the smooth distributions the Limber
integrals consume (both settable per likelihood in its yaml, both
defaulting to the historical behavior):

  photoz_interpolation_type - the stage-1 interpolant of the two-stage
      n(z) scheme: 0 = cubic spline (the default), 1 = linear,
      2+ = Steffen monotone (no overshoot: n(z) can never ring below
      zero around a sharp feature in the table).
  photoz_zmid_convention - how the z column of the n(z) file is read:
      0 = Z_LOW (the default; the column holds left bin edges, so the
      tabulated value belongs at the cell center z + dz/2), 1 = Z_MID
      (the column holds the sample points themselves). The two
      readings differ by a rigid dz/2 shift of every distribution,
      which is percent-level in cosmic shear.

This test evaluates the frozen cosmic-shear fiducial under five
settings IN ONE PROCESS: the default, each alternative, and the
default again. For each alternative it measures

    delta chi2 = delta^T C^-1 delta,
    delta = dv(alternative) - dv(default),

with C^-1 the masked inverse covariance from the compiled interface -
the same second-order construction the CFASTPT-vs-FASTPT sweep uses
(the chi2 the alternative would score against a dataset whose data
vector IS the default prediction), which never rides the slope of the
distance to the shipped data.

Running everything in one process is the point, not a convenience:
the n(z) table caches inside the compiled interface must notice a
runtime flag change and rebuild. A stale cache would make every
delta exactly zero (the alternatives assert delta chi2 > 0), and the
final return to the default must reproduce the first vector (the
round-trip assertion), so a cache that over- or under-invalidates
fails loudly here.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/test_photoz_conventions.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import shutil
import sys
import tempfile
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

EXAMPLE = "example1"

# (report tag, photoz_interpolation_type, photoz_zmid_convention)
SETTINGS = (
    ("cspline/Z_LOW (default)", 0, 0),
    ("linear", 1, 0),
    ("steffen", 2, 0),
    ("Z_MID", 0, 1),
    ("default again (round trip)", 0, 0),
)

# The alternatives must be SEEN (a stale n(z) cache would give exactly
# zero); the floors are orders of magnitude below the measured deltas,
# so they only catch a dead flag, never normal numerical drift.
DCHI2_FLOORS = {"linear": 1.0e-8, "steffen": 1.0e-8, "Z_MID": 1.0e-2}


class TestPhotozConventions(unittest.TestCase):
    """The five-setting sweep, sharing one frozen-state verification."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_photoz_conventions(self):
        import numpy as np
        import cosmolike_desy1xplanck_interface as ci

        vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_")
        self.addCleanup(shutil.rmtree, vectors_dir, ignore_errors=True)

        results = {}
        icov = None
        for tag, interp, zmid in SETTINGS:
            print(f"  building model ({EXAMPLE}, NLA, {tag}) ...",
                  flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            likelihood_block = \
                info["likelihood"][u.EXAMPLES[EXAMPLE]["likelihood"]]
            likelihood_block["photoz_interpolation_type"] = interp
            likelihood_block["photoz_zmid_convention"] = zmid
            vector_path = os.path.join(
                vectors_dir, f"dv_{interp}_{zmid}.modelvector")
            likelihood_block["print_datavector"] = True
            likelihood_block["print_datavector_file"] = vector_path
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            chi2 = u.evaluate_chi2(model, point)
            if not os.path.isfile(vector_path):
                raise RuntimeError(
                    f"print_datavector wrote no file at {vector_path}")
            results[tag] = (chi2, u._load_datavector(vector_path))
            if icov is None:
                # all five settings share one dataset (mask, covariance),
                # so the masked inverse covariance of the first build
                # serves every comparison
                icov = np.array(ci.get_inv_cov_masked())

        chi2_default, dv_default = results[SETTINGS[0][0]]

        print(f"\n  chi2 report ({EXAMPLE}, NLA):")
        print(f"    default: chi2 = {chi2_default:.6f} "
              f"(frozen reference {self.reference['example1_nla']:.6f})")
        for tag in ("linear", "steffen", "Z_MID"):
            chi2, dv = results[tag]
            delta = dv - dv_default
            dchi2 = float(delta @ icov @ delta)
            print(f"    {tag:8s}: chi2 = {chi2:.6f}, "
                  f"delta^T C^-1 delta vs default = {dchi2:.6e}")
            self.assertGreater(
                dchi2, DCHI2_FLOORS[tag],
                f"{tag}: the flag change was not seen by the n(z) "
                "cache (delta chi2 at or below the dead-flag floor)")

        # the default must agree with the frozen reference exactly as
        # the standard chi2 drift test demands
        self.assertLess(
            abs(chi2_default - self.reference["example1_nla"]),
            u.CHI2_TOLERANCE)

        # round trip: after all the flips, the default settings must
        # reproduce the first vector identically - the cache rebuilt
        # back to the same state
        _, dv_return = results[SETTINGS[-1][0]]
        self.assertTrue(
            np.array_equal(dv_return, dv_default),
            "returning to the default settings did not reproduce the "
            "default data vector bit for bit; the n(z) cache did not "
            "rebuild cleanly on the way back")


if __name__ == "__main__":
    unittest.main(verbosity=2)
