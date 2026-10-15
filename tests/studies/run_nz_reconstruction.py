"""Explicit n(z) sensitivity study; this file is not collected by pytest.

Activate Cocoa first. Use --output with a new directory; the full refined
comparison is expensive and is never part of the usual regression suite.
"""

import os
from pathlib import Path
import sys

# Set BLAS limits before importing NumPy or the compiled interface.
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "8")
root = Path(os.environ["ROOTDIR"])
sys.path.insert(0, str(root / "cobaya"))
sys.path.insert(0, str(root / "external_modules/code/cosmolike_core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cocoa_test_utils as harness
import cosmolike_desy1xplanck_interface as native
from cocoa_nz_study import main


if __name__ == "__main__":
    main(harness=harness, native=native, project="desy1xplanck")
