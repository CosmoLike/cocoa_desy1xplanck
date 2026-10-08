"""Locate the project interface for the optional covariance test sector.

pytest reads this conftest.py before it runs the tests of this folder.
It puts the project's interface/ folder on the import path and skips
every test here when the compiled interface was built without
covariance generation (the default build; see the project README).
"""

from pathlib import Path
import sys

# parents[2] of this file's path is the project folder
# (projects/desy1xplanck); the / operator of pathlib joins path pieces
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

import pytest
import cosmolike_desy1xplanck_interface as ci


# A fixture is a function pytest runs around the tests; scope="session"
# runs it once for the whole pytest run, and autouse=True applies it to
# every test without the tests asking for it.
@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip this sector when covariance generation was intentionally omitted.

    A build with covariance generation sets the attribute has_covariance
    of the compiled module; getattr returns False when it is absent.

    Returns:
      nothing; pytest.skip marks every test of the sector as skipped,
      with the message telling the user how to enable the build.
    """
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
