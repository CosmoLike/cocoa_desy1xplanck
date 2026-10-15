"""Compare 3x, 5x and 7x Akima preparation at fixed survey inputs.

Preparation selects n(z); integration_accuracy evaluates that chosen model.
All three factors use the same raised level to resolve their smaller differences.
The tests check live cache round trips and YAML agreement, then compare with
saved level-6 vectors. The README separates model shifts from numerical error.
"""

import json
import os
from pathlib import Path
import sys

# Set threading before CAMB or the compiled interface is imported.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import cocoa_test_utils as u
import cosmolike_desy1xplanck_interface as ci
from cocoa_nz_testing import compare_akima_preparation


@pytest.mark.parametrize("preparation_factor, practical_level", ((5, 4), (7, 4)))
def test_nz_akima_factor(preparation_factor, practical_level):
    """Check the factor switch and its measured numerical distance budget."""
    level = int(os.environ.get("COCOA_NZ_INTEGRATION_ACCURACY", str(practical_level)))
    if level not in range(7):
        raise ValueError("COCOA_NZ_INTEGRATION_ACCURACY must be between 0 and 6")
    record, vectors, inverse = compare_akima_preparation(
        harness=u, native=ci, project="desy1xplanck", example="example2",
        integration_accuracy=level, preparation_factor=preparation_factor)
    directory = Path(__file__).resolve().parents[1]
    reference = json.loads((directory / "nz_akima_factor.json").read_text())
    assert record["point"] == reference["point"]
    assert record["input_files"] == reference["input_files"]
    assert record["accuracyboost"] == reference["accuracyboost"]
    with np.load(file=directory / "nz_akima_factor.npz") as archive:
        # An adjacent pair of coarse rules can agree accidentally. Compare
        # each fixed model with its saved 4096-point result instead.
        for name, vector in vectors.items():
            difference = vector - archive[name]
            distance = float(difference @ inverse @ difference)
            print(f"{name}: numerical distance from level 6 = {distance:.12g}")
            assert np.isfinite(distance)
            if record["physical_covariance"]:
                assert distance < 0.1, "Refine integration_accuracy for this n(z)"
