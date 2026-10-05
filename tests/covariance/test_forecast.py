"""Check this project's galaxy/shear covariance interface and catalog inputs.

Run separately from tests/data_vector; the shared check uses small numerical
settings and a measured subset, so it does not certify survey convergence.
"""

from pathlib import Path
import sys

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))
sys.path.insert(0, str(project/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_desy1xplanck_interface as ci
import desy1xplanck_covariance as survey


def test_forecast_adapter(tmp_path):
    """Real and Fourier components repeat at one/eight threads and save intact."""
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(1500, 600), directory=tmp_path,
    )
