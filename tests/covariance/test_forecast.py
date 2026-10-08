"""Check this project's galaxy/shear covariance interface and catalog inputs.

The check itself is shared by every project (check_project_forecast in
external_modules/code/cosmolike_core/cocoa_covariance_testing.py). For
desy1xplanck it checks the measured layout (1500 real-space entries:
cosmic shear, galaxy-galaxy lensing and clustering in 30 angular bins;
600 Fourier entries in 15 bands) and that refining the accuracy keeps
the measurement bins fixed. Then, for three rows (one shear, one
galaxy-galaxy lensing and one clustering row) on reduced grids, it
checks that the G, SSC, cNG and total matrices are finite and
symmetric, agree between the notebook and the command-line bindings,
repeat bit for bit at one and eight OpenMP threads, that the total is
the sum of the parts and positive definite, and that the saved archive
reads back intact.

Run separately from tests/data_vector; the shared check uses small numerical
settings and a measured subset, so it does not certify survey convergence.
"""

from pathlib import Path
import sys

# parents[2] of this file's path is the project folder, and its
# parents[1] the Cocoa/ folder: the shared cosmolike code, the compiled
# interface and the survey module of covariance/ go on the import path
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))
sys.path.insert(0, str(project/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_desy1xplanck_interface as ci
import desy1xplanck_covariance as survey


def test_forecast_adapter(tmp_path):
    """Real and Fourier components repeat at one/eight threads and save intact.

    Arguments:
      tmp_path = a temporary folder that pytest creates for this test
                 (a fixture named in the argument list); the check writes
                 its archives there.

    Returns:
      nothing; a failed assertion inside check_project_forecast fails
      the test.
    """
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(1500, 600), directory=tmp_path,
    )
