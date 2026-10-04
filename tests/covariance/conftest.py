"""Locate the project interface for the optional covariance test sector."""

from pathlib import Path
import sys

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

import pytest
import cosmolike_desy1xplanck_interface as ci


@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip this sector when covariance generation was intentionally omitted."""
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
