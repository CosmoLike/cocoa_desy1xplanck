"""Locate the project interface for the optional covariance test sector.

pytest reads a file named conftest.py before it collects the tests of
the same folder, so the code here runs first for every test of
tests/covariance. It puts interface/ (the folder of the compiled
module) first on the import path and defines one fixture that skips
the whole folder when the library was compiled without its covariance
code.
"""

from pathlib import Path
import sys

# parents[2] of this file is the project folder; its interface/ folder
# holds the compiled module. Resolving from this file keeps the
# documented commands working from Cocoa/ and from tests/.
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

import pytest
import cosmolike_desy1xplanck_interface as ci


# autouse=True applies the fixture to every test of this folder without
# the tests naming it; scope="session" runs it once per pytest run
@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip this sector when covariance generation was intentionally omitted.

    The compiled module exposes has_covariance = True only when it was
    built with the covariance code; otherwise every covariance test is
    skipped with the rebuild instruction.
    """
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
