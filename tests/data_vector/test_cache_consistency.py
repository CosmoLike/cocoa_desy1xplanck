"""Unit test: sector-wise cache invalidation (the parameter ladder).

cosmolike caches every expensive stage behind its own key: cosmology
(distances, growth, power-spectrum tables), intrinsic alignment
(FAST-PT tables under TATT), photo-z shifts (n(z) splines, lens
efficiencies), and the shear calibrations (a pure data-vector
rescale). A partial-invalidation bug - one sector's update path
failing to rebuild a static another sector consumes - produces
silently wrong data vectors only in MIXED update sequences, which the
per-point suites never exercise.

The test walks a deterministic ladder IN ONE PROCESS, evaluating the
model after every step (each sector's later steps keep the earlier
sectors at their last values, so the ladder ends at one well-defined
point):

    3 x cosmology-only steps   (omegam, H0, As_1e9)
    3 x IA-only steps          (A1; under TATT also A2 and BTA)
    3 x source-photo-z steps   (every DZ_S shift)
    3 x lens-photo-z steps     (every DZ_L shift)
    3 x shear-calibration steps (every M)

(a sector the configuration samples no parameters in drops out of the
ladder: separate DZ_L shifts when the lenses ARE the source sample, IA
amplitudes where the configuration fixes them)

It records the final data vector, then evaluates one SCRAMBLE point
(every sector moved at once, galaxy bias included; the chi2 is
discarded) and returns to the ladder's final point: the pipeline must
reproduce the recorded vector bit for bit. A second model instance
walks the MIRRORED ladder (M -> DZ_L -> DZ_S -> IA -> cosmology) to
the same final point: the answer must depend on the point, never on
the invalidation history.

Assertions, in each intrinsic-alignment model (NLA and TATT - the
TATT ladder exercises the FAST-PT rebuild machinery NLA never
touches):
  1. every ladder step changes the data vector (a dead sector flag
     would pass the later checks vacuously);
  2. each M-only step rescales the masked vector by the analytic
     (1+m_i)(1+m_j) block factors to 1e-12 relative - cosmic shear by
     both bins' factors, gamma_t by the source factor, w by nothing;
  3. a no-op update (re-sending the current M values) leaves the
     vector bitwise unchanged;
  4. after the scramble, returning to the ladder's final point
     reproduces the recorded vector and chi2 bit for bit;
  5. the mirrored-order instance lands on the same final vector bit
     for bit.

Every evaluation forces a full recomputation (cobaya's cache is
bypassed), so each assertion tests cosmolike's own invalidation, not
cobaya's memoization.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_cache_consistency.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example2"

# Sector membership by sampled-parameter name. Every sampled parameter
# must fall in exactly one sector (checked at run time); the last two
# sectors move only in the scramble step.
SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("other", re.compile(r".")),
)

# Ladder phases (sector order of the forward walk) and the per-step
# offsets: parameter value = fiducial + step * delta, deterministic.
PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m")
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
}
NSTEP = 3
SCRAMBLE_STEP = 4  # every sector at step 4, bias included
RESCALE_RTOL = 1.0e-12


def _sector_of(name):
    for sector, pat in SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _deltas_for(sector, names):
    """{parameter: per-step delta} for this sector's sampled names."""
    table = DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


class TestCacheConsistency(unittest.TestCase):
    """Sector-ladder cache-invalidation check on the frozen fiducial."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def _point_at(self, fid, steps):
        """The ladder point with each sector at its given step count."""
        point = dict(fid)
        for sector, step in steps.items():
            for n, d in self.sector_deltas[sector].items():
                point[n] = fid[n] + step * d
        return point

    def _mpairs(self, like, np_):
        """(i, j) source-bin shear-calibration factors per masked-vector
        entry: cosmic shear scales by both bins, gamma_t (and the ks
        cross where present) by the source bin, clustering, gk and kk
        by nothing. Fourier data vectors use ncl per block, real ones
        ntheta; the xi_pm split doubles the shear block in real space."""
        import cosmolike_desy1xplanck_interface as ci
        real = hasattr(ci, "compute_data_vector_3x2pt_real_sizes")
        sizes = [int(x) for x in
                 (ci.compute_data_vector_3x2pt_real_sizes() if real else
                  ci.compute_data_vector_3x2pt_fourier_sizes())]
        nlen = int(like.ntheta) if real else int(like.ncl)
        nsrc = int(like.source_ntomo)
        sspairs = [(i, j) for i in range(nsrc) for j in range(i, nsrc)]
        excluded = {(int(a), int(b)) for a, b in
                    (getattr(like, "ggl_exclude", None) or [])}
        gglpairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                    for zs in range(nsrc) if (zl, zs) not in excluded]
        fac = np_.zeros((sum(sizes), 2), dtype=int) - 1
        k = 0
        ssrep = sspairs + sspairs if real else sspairs  # xi_plus + xi_minus
        for (i, j) in ssrep:
            for t in range(nlen):
                fac[k] = (i, j)
                k += 1
        for (zl, zs) in gglpairs:
            for t in range(nlen):
                fac[k] = (-1, zs)
                k += 1
        k = sizes[0] + sizes[1] + sizes[2]  # skip clustering
        if len(sizes) > 3:  # 6x2pt: gk (no factor), ks (source), kk (none)
            k += sizes[3]
            for zs in range(nsrc):
                for t in range(nlen):
                    fac[k] = (-1, zs)
                    k += 1
        return fac

    def _run_ladder(self, tatt):
        import numpy as np
        import cosmolike_desy1xplanck_interface as ci

        name = u.EXAMPLES[EXAMPLE]["likelihood"]
        results = {}
        for order in ("forward", "mirrored"):
            info = u.load_frozen_info(EXAMPLE, tatt=tatt)
            model = u.make_model(info)
            fid = dict(u.build_point(model, EXAMPLE, tatt=tatt))
            self.sector_deltas = {
                s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                for s, _ in SECTORS}
            for s in ("cosmo", "dz_source", "m"):
                self.assertTrue(self.sector_deltas[s],
                                f"no sampled parameters in sector {s}")
            # a sector nothing samples drops out of the ladder: lenses
            # that ARE the source sample carry no separate DZ_L shifts,
            # and roman_kl fixes every IA amplitude
            active = tuple(s for s in PHASES if self.sector_deltas[s])

            phases = active if order == "forward" else tuple(reversed(active))
            steps = {s: 0 for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, steps))
            prev = np.array(ci.compute_data_vector_masked())
            mfac = self._mpairs(model.likelihood[name], np)

            for sector in phases:
                for r in range(1, NSTEP + 1):
                    m_prev = {n: fid[n] + steps["m"] * d
                              for n, d in self.sector_deltas["m"].items()}
                    steps[sector] = r
                    point = self._point_at(fid, steps)
                    u.evaluate_chi2(model, point)
                    dv = np.array(ci.compute_data_vector_masked())
                    self.assertFalse(
                        np.array_equal(dv, prev),
                        f"{order}: {sector} step {r} left the data vector "
                        "unchanged (dead sector flag or stale cache)")
                    if sector == "m":
                        m_now = {n: point[n]
                                 for n in self.sector_deltas["m"]}
                        mp = sorted(m_prev)  # M1..M5 in bin order
                        ratio = np.ones(dv.size)
                        for k in range(dv.size):
                            i, j = mfac[k]
                            if j >= 0:
                                ratio[k] *= ((1 + m_now[mp[j]]) /
                                             (1 + m_prev[mp[j]]))
                            if i >= 0:
                                ratio[k] *= ((1 + m_now[mp[i]]) /
                                             (1 + m_prev[mp[i]]))
                        nz = prev != 0
                        rel = np.abs(dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                        self.assertLess(
                            rel.max(), RESCALE_RTOL,
                            f"{order}: M step {r} is not the analytic "
                            f"(1+m_i)(1+m_j) rescale (max {rel.max():.2e})")
                    prev = dv

            final_point = self._point_at(fid, steps)
            final_chi2 = u.evaluate_chi2(model, final_point)
            final_dv = np.array(ci.compute_data_vector_masked())

            # no-op probe: identical point again, bitwise
            u.evaluate_chi2(model, dict(final_point))
            self.assertTrue(
                np.array_equal(np.array(ci.compute_data_vector_masked()),
                               final_dv),
                f"{order}: a no-op re-evaluation changed the data vector")

            # scramble: every sector at once, bias included
            scr = {s: SCRAMBLE_STEP for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, scr))

            # return: the ladder's final point must reproduce bitwise
            back_chi2 = u.evaluate_chi2(model, final_point)
            back_dv = np.array(ci.compute_data_vector_masked())
            self.assertTrue(
                np.array_equal(back_dv, final_dv),
                f"{order}: returning after the scramble did not reproduce "
                "the data vector bit for bit (stale sector cache)")
            self.assertEqual(
                back_chi2, final_chi2,
                f"{order}: chi2 after the scramble return differs")
            results[order] = final_dv
            print(f"  {order} ladder ({'TATT' if tatt else 'NLA'}): "
                  f"final chi2 = {final_chi2:.6f}", flush=True)

        self.assertTrue(
            np.array_equal(results["forward"], results["mirrored"]),
            "the mirrored-order ladder landed on a different data vector: "
            "the answer depends on the invalidation history")

    def test_cache_consistency_nla(self):
        self._run_ladder(tatt=False)

    def test_cache_consistency_tatt(self):
        self._run_ladder(tatt=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
