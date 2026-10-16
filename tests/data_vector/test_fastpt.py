"""Unit tests 15-17: cfastpt vs python FAST-PT, compared directly.

Cosmolike offers two implementations of the perturbation-theory
integrals that the TATT intrinsic-alignment model needs: cfastpt, a C
implementation built into the cosmolike interface (`IA_code: 0`, the
default), and the python FAST-PT package used through the fastpt
theory block (`IA_code: 1`).

15. example1 (cosmic shear): the same 30 hard-coded points across
     the intrinsic-alignment prior (FASTPT_COMPARISON_POINTS: 20 drawn
     across the prior boxes plus a one-parameter-at-a-time family;
     cosmology fixed at the frozen fiducial), evaluated in three
     blocks: with cfastpt, with FASTPT at the pass configuration
     (FASTPT_LOW_SETTINGS, written out in cocoa_test_utils), and with
     FASTPT at the doubled boosts (FASTPT_HIGH_SETTINGS). Every block
     prints its theory vector at every point, and the CFASTPT vector
     is the fiducial of that point: its own chi2 against it is zero by
     construction, so the pass rule is the chi2 of the FASTPT(low)
     vector against it (delta^T C^-1 delta, a pure second-order
     deviation; a chi2 difference against the shipped data would
     follow the slope of the chi2 instead). FASTPT(high)'s deviation is
     printed as the advisory FAST-PT grid response. Each block runs in
     its own worker subprocess, so no cache survives from one block to
     the next; inside a block the shared cosmology makes CAMB run once
     and the 30 points cheap.
16. example2 (6x2pt): the same three-block sweep as test 15 on the
     6x2pt likelihood, so the TATT terms are also scored inside
     galaxy-galaxy lensing and the difference is weighted by the
     6x2pt masked inverse covariance. The frozen configuration fixes
     the one-loop bias amplitudes (DES_B2_*) at zero, so the
     one-loop galaxy-bias tables both implementations compute
     multiply by zero here, and the CMB-lensing cross-correlations
     carry only the linear (A1) alignment term, no second-order
     tables: this sweep compares the intrinsic-alignment tables
     only, on the wider data vector.
17. example2_2x2pt (2x2pt): the same sweep on the 2x2pt likelihood
     (galaxy clustering plus galaxy-galaxy lensing, everything else
     dropped). Clustering carries no intrinsic alignment, so here
     the TATT tables are scored through galaxy-galaxy lensing
     alone, under the 2x2pt masked inverse covariance; DES_B2_*
     stays zero as in test 16.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_fastpt.py

--high=1 repeats every block at the pushed camb/cosmolike settings
of the low-vs-high accuracy checks; the full comparison is one run
without the option and one with it.

The tests also read the --mask option (see conftest.py):
--mask=frozen (the default) keeps the shipped 6x2pt scale-cut mask
of the frozen configurations, and --mask=ones keeps every data point
(no scale cuts), the strictest comparison; the 0.2 pass rule applies
unchanged. The 6x2pt sweep (test 16) aborts under ones: with every
point unmasked the shipped 6x2pt covariance is not positive
definite and cosmolike refuses it (see tests/data_vector/README.md):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_fastpt.py --mask=ones

The design and the point values are shared with lsst_y1's tests
15-17; see that project's tests/README.md for the full discussion.
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factories (cosmolike_core/cocoa_testing_suites.py) build
# the classes; assigning each return value to the usual module-level
# name is what makes pytest collect it here, with unchanged test ids.
TestCfastptVsFastptSweep = u.fastpt_sweep_suite(
    u, name="TestCfastptVsFastptSweep",
    cases=(
        ('test_x15_cfastpt_vs_fastpt_sweep', 15, 'example1', 'example1 (cosmic shear'),
        ('test_x16_cfastpt_vs_fastpt_sweep_6x2pt', 16, 'example2', 'example2 (6x2pt'),
        ('test_x17_cfastpt_vs_fastpt_sweep_2x2pt', 17, 'example2_2x2pt', 'example2_2x2pt (2x2pt'),
    ))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
