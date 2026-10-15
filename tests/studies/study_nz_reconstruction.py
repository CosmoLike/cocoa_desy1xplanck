"""Opt-in test of n(z) reconstruction choices and numerical accuracy.

Run this file explicitly with pytest. Its study_ prefix deliberately does
not match pytest's default test_*.py or *_test.py discovery patterns, so an
ordinary regression run does not repeat this expensive scientific study.

The child script checks finite predictions, unchanged inputs/covariance,
cache round trips and compiled-reader cell integrals. It reports model
separations without imposing a frozen vector or an expected Delta chi2.
"""
from pathlib import Path
import subprocess
import sys


def test_nz_reconstruction_study(tmp_path):
    """Compute fresh original/3/5/7 vectors, cell areas and the two-panel plot.

    Arguments:
      tmp_path = pytest-provided empty directory belonging to this test.

    Returns:
      None. The study writes JSON records, NPZ arrays and a PNG below the
      printed output path. A failed scientific check or subprocess fails
      this test; no reference measurements are overwritten.

    A separate Python process owns the mutable CosmoLike state. This keeps
    earlier tests' cosmology and reconstruction caches out of the study.
    """
    script = Path(__file__).with_name("run_nz_reconstruction.py")
    output = tmp_path / "nz_reconstruction"
    command = [
        sys.executable,
        str(script),
        "--output",
        str(output),
    ]
    subprocess.run(args=command, check=True)
    print(f"Saved n(z) study measurements and figure: {output}", flush=True)
