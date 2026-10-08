"""Unit test: non-Limber galaxy-galaxy lensing (adopt_limber_gs).

The galaxy-galaxy lensing (ggl) spectrum C_l^gs enters the data vector
through gamma_t(theta) in real space and directly in Fourier space. The
likelihood yaml key adopt_limber_gs chooses how it is computed:

  adopt_limber_gs: 1 (the default) - Limber approximation at every
      multipole.
  adopt_limber_gs: 0 - below l = 150 the exact projection, computed by
      cosmolike's C_gs_tomo with the split of Fang, Krause, Eifler &
      MacCrann (arXiv:1911.11947): an FFTLog integral of the linear
      power spectrum plus, in Limber, what linear theory misses. In
      Fourier space each band center takes the Limber value plus the
      non-Limber correction interpolated between integer multipoles.

ggl defaults to Limber because its lensing kernel is broad (galaxy
clustering has its own key, adopt_limber_gg; see
test_nonlimber_gg.py). The Limber approximation fails at low l for
the lens-source pairs whose kernels overlap in redshift (lens bin =
source bin, or the source bin in front of the lens bin, where the
signal is the intrinsic alignment of the sources times the lens
density). This test measures what the Limber default costs.

It evaluates the frozen 6x2pt fiducial (NLA) three times in one
process (so the caches of the compiled library must follow the flag):
Limber, non-Limber, Limber again, and computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(non-Limber) - dv(Limber),

with C^-1 the masked inverse covariance: the chi2 the Limber model
would score against a data set generated with non-Limber ggl. It prints
the total and the contribution of each lens-source pair (the pair's own
block of delta, cross-covariance with other pairs ignored).

Assertions:
  1. delta chi2 is above a dead-flag floor: the flag reaches the C code
     and the ggl cache notices the change (a stale cache gives zero);
  2. only ggl entries change: cosmic shear, clustering, and every other
     block are bitwise equal between the two evaluations;
  3. switching back to Limber reproduces the first data vector bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 5%: a change in the non-Limber code, the
     kernels, or the covariance shows up here;
  5. the Limber evaluation reproduces the frozen reference chi2
     (checked last, so a stale snapshot cannot hide checks 1-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_nonlimber_ggl.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process. "4"
# is REQUIRED_OMP_THREADS of cocoa_testing, the count of every worker
# subprocess: a race check needs more than one thread.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time
import unittest

# The harness stays in the parent tests/ folder (dirname applied twice to
# this file's absolute path). Add it explicitly so direct execution and
# worker processes resolve this project's stored inputs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example2"
REFERENCE_KEY = "example2_nla"

# (report tag, adopt_limber_gs)
SETTINGS = (
    ("Limber (default)", 1),
    ("non-Limber", 0),
    ("Limber again (round trip)", 1),
)

# A stale ggl cache gives delta chi2 = 0 exactly; the floor is orders of
# magnitude below the measured value, so it only catches a dead flag.
DCHI2_FLOOR = 1.0e-6

# The delta chi2 of this comparison as measured on macOS (arm64), and the
# relative band assertion 4 allows around it; a deliberate change of the
# non-Limber code, the kernels or the covariance requires measuring it
# again.
DCHI2_MEASURED = 0.003869
DCHI2_RTOL = 0.05


class TestNonLimberGGL(unittest.TestCase):
    """Limber vs non-Limber ggl on the frozen fiducial."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_nonlimber_ggl(self):
        """Build the model three times, compare the vectors, assert 1-5.

        The five assertions are listed in the module docstring; the
        report prints the two chi2 values, delta^T C^-1 delta and the
        contribution of each lens-source pair.
        """
        import numpy as np
        import cosmolike_desy1xplanck_interface as ci

        vectors = {}
        chi2s = {}
        icov = None
        sizes = None
        nlen = None
        pairs = None
        for tag, flag in SETTINGS:
            print(f"  building model ({EXAMPLE}, NLA, {tag}) ...",
                  flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            name = u.EXAMPLES[EXAMPLE]["likelihood"]
            info["likelihood"][name]["adopt_limber_gs"] = flag
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            start = time.perf_counter()
            chi2s[tag] = u.evaluate_chi2(model, point)
            elapsed = time.perf_counter() - start
            print(f"    chi2 = {chi2s[tag]:.6f}  ({elapsed:.1f} s, "
                  "first evaluation of the model)", flush=True)
            # full-precision model vector (the printed one keeps 9 digits)
            vectors[tag] = np.array(ci.compute_data_vector_masked())
            if icov is None:
                # all settings share one data set (mask, covariance), so
                # the first build's masked inverse covariance serves all
                icov = np.array(ci.get_inv_cov_masked())
                like = model.likelihood[name]
                if hasattr(ci, "compute_data_vector_3x2pt_real_sizes"):
                    sizes = ci.compute_data_vector_3x2pt_real_sizes()
                    nlen = int(like.ntheta)
                else:
                    sizes = ci.compute_data_vector_3x2pt_fourier_sizes()
                    nlen = int(like.ncl)
                # the ggl pairs in data-vector order: lens-major, the
                # pairs listed in the yaml key ggl_exclude left out
                excluded = {(int(zl), int(zs)) for zl, zs in
                            (getattr(like, "ggl_exclude", None) or [])}
                pairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                                  for zs in range(int(like.source_ntomo))
                                  if (zl, zs) not in excluded]

        dv_limber = vectors[SETTINGS[0][0]]
        dv_nonlimber = vectors[SETTINGS[1][0]]
        delta = dv_nonlimber - dv_limber
        # @ is numpy's matrix product: delta @ icov @ delta = delta^T C^-1
        # delta
        dchi2 = float(delta @ icov @ delta)

        # the ggl block follows the cosmic shear block in every probe
        # combination (3x2pt, 2x2pt, 6x2pt): entries [ggl0, ggl1)
        ggl0 = int(sizes[0])
        ggl1 = ggl0 + int(sizes[1])
        npairs = int(sizes[1]) // nlen

        print(f"\n  delta chi2 report ({EXAMPLE}, NLA):")
        print(f"    Limber:     chi2 = {chi2s[SETTINGS[0][0]]:.6f} "
              f"(frozen reference {self.reference[REFERENCE_KEY]:.6f})")
        print(f"    non-Limber: chi2 = {chi2s[SETTINGS[1][0]]:.6f}")
        print(f"    delta^T C^-1 delta = {dchi2:.4f} "
              f"(measured {DCHI2_MEASURED:.4f})")
        print("    per lens-source pair (the pair's block alone):")
        rows = []
        for p in range(npairs):
            block = np.zeros_like(delta)
            sl = slice(ggl0 + p*nlen, ggl0 + (p + 1)*nlen)
            block[sl] = delta[sl]
            rows.append((float(block @ icov @ block), p))
        # largest contribution first (the tuples sort by their first
        # entry); the list stops below 0.1% of the total. The label names
        # the bins when the pair list matches the block count, otherwise
        # the pair index.
        for contribution, p in sorted(rows, reverse=True):
            if contribution < 1.0e-3*max(dchi2, DCHI2_FLOOR):
                break
            label = (f"(lens {pairs[p][0]}, source {pairs[p][1]})"
                     if len(pairs) == npairs else f"pair {p}")
            print(f"      {label:24s} {contribution:.4f}")

        self.assertGreater(
            dchi2, DCHI2_FLOOR,
            "non-Limber ggl did not change the data vector: the "
            "adopt_limber_gs flag did not reach the C code, or the ggl "
            "cache did not rebuild")

        outside = np.concatenate((delta[:ggl0], delta[ggl1:]))
        self.assertTrue(
            np.all(outside == 0.0),
            "entries outside the ggl block changed with adopt_limber_gs")

        self.assertTrue(
            np.array_equal(vectors[SETTINGS[-1][0]], dv_limber),
            "returning to Limber did not reproduce the first data vector "
            "bit for bit; the ggl cache did not rebuild cleanly")

        self.assertLess(
            abs(dchi2/DCHI2_MEASURED - 1.0), DCHI2_RTOL,
            f"delta chi2 = {dchi2:.4f} differs from the measured "
            f"{DCHI2_MEASURED:.4f} by more than {DCHI2_RTOL:.0%}")

        # last, so that a stale frozen snapshot does not hide the four
        # checks above
        self.assertLess(
            abs(chi2s[SETTINGS[0][0]] - self.reference[REFERENCE_KEY]),
            u.CHI2_TOLERANCE,
            f"{SETTINGS[0][0]}: chi2 = {chi2s[SETTINGS[0][0]]:.6f} vs frozen "
            f"reference {self.reference[REFERENCE_KEY]:.6f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
