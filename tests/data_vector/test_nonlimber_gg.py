"""Unit test: non-Limber galaxy clustering (adopt_limber_gg).

The galaxy clustering (gg) spectrum C_l^gg enters the data vector
through w(theta) in real space and directly in Fourier space. The
likelihood yaml key adopt_limber_gg chooses how it is computed:

  adopt_limber_gg: 0 - below l = 150 the exact projection, computed by
      cosmolike's C_cl_tomo with the split of Fang, Krause, Eifler &
      MacCrann (arXiv:1911.11947): an FFTLog integral of the linear
      power spectrum plus, in Limber, what linear theory misses. In
      Fourier space each band center takes the Limber value plus the
      non-Limber correction interpolated between integer multipoles.
  adopt_limber_gg: 1 - Limber approximation at every multipole.

This project's default is adopt_limber_gg: 0 (non-Limber). The
lens galaxy redshift distributions are narrow, so the Limber
approximation fails at low l for the clustering auto spectra; this
test measures by how much.

It evaluates the frozen 6x2pt fiducial (NLA) three times in one
process (so the caches of the compiled library must follow the flag):
the default, the other setting, the default again, and computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(non-Limber) - dv(Limber),

with C^-1 the masked inverse covariance: the chi2 a Limber model would
score against a data set generated with non-Limber clustering. It
prints the total and the contribution of each lens bin (the bin's own
block of delta, cross-covariance with other bins ignored).

Assertions:
  1. delta chi2 is above a dead-flag floor: the flag reaches the C code
     and the clustering cache notices the change (a stale cache gives
     zero);
  2. only clustering entries change: cosmic shear, galaxy-galaxy
     lensing, and every other block are bitwise equal between the two
     evaluations;
  3. switching back to the default reproduces the first data vector
     bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 5%: a change in the non-Limber code, the
     kernels, or the covariance shows up here;
  5. the default evaluation reproduces the frozen reference chi2
     (checked last, so a stale snapshot cannot hide checks 1-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/desy1xplanck/tests/data_vector/test_nonlimber_gg.py
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

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here, with unchanged test ids.
TestNonLimberGG = u.nonlimber_suite(
    u, name="TestNonLimberGG", kind="gg",
    measured=6.613, default=0)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
