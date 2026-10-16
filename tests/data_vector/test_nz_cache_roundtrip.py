"""Live n(z) selection changes behave like any other cache invalidation.

Flipping the z-column reading (Z_LOW left edges, Z_MID sample points),
the reconstruction (midpoint curve, histogram plus Akima), the
preparation factor or the stage-1 interpolant must take effect on the
very next query, move the fiducial means together with the curves, and
restore every value bit for bit when the original selection returns.

These are cache-behavior checks, not accuracy checks: the compiled
production readers and the stored fiducial means are compared bitwise
across selection round trips on synthetic histograms, with no
cosmology calculation involved.
"""
import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

# start_cocoa.sh puts a project's interface folder on the import path
# only when that project is enabled in set_installation_options.sh.
# This cache check must run for any compiled project, so resolve the
# folder from this file's own location (tests/data_vector -> interface).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "interface"))

import cosmolike_desy1xplanck_interface as ci

# How many trailing entries of state() hold fiducial means (two lens
# bins, then two source bins); the rest are dense reader samples.
NMEANS = 4


def compiled_readers():
    """The production C readers and mean accessors inside the module."""
    library = ctypes.CDLL(str(Path(ci.__file__).resolve()))
    for name in ("nz_source_photoz", "nz_lens_photoz"):
        getattr(library, name).argtypes = [ctypes.c_double, ctypes.c_int]
        getattr(library, name).restype = ctypes.c_double
    for name in ("zmean", "zmean_source"):
        getattr(library, name).argtypes = [ctypes.c_int]
        getattr(library, name).restype = ctypes.c_double
    return library


def write_histogram(tmp_path, irregular):
    """A two-bin synthetic histogram file, uniform or irregular in z."""
    z = np.arange(9, dtype=float) * .1 + .05
    if irregular:
        z = np.array([.05, .11, .18, .24, .39, .42, .61, .76, .85])
    heights = np.array([0., .2, .7, 1., .4, .6, .1, 0., 0.])
    table = np.column_stack((z, heights, heights[::-1]))
    filename = tmp_path / "histogram.nz"
    np.savetxt(fname=filename, X=table)
    return filename


def load_sample(filename):
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=str(filename), lens_ntomo=2,
        source_multihisto_file=str(filename), source_ntomo=2)


def state(library):
    """Everything one n(z) selection owns: curves plus fiducial means.

    The dense samples watch the reconstructed curves; the means watch
    the derived state only the sample loaders compute (the lens means
    are the pivot of the photo-z stretch). A selection change that
    rebuilds one without the other is exactly the staleness this test
    exists to refuse.
    """
    query = np.linspace(start=.025, stop=1.0, num=2001)
    values = []
    for sample in ("source", "lens"):
        reader = getattr(library, "nz_" + sample + "_photoz")
        for tomo in range(2):
            values.append(np.fromiter(
                (reader(float(x), tomo) for x in query),
                dtype=float, count=query.size))
    means = [library.zmean(tomo) for tomo in range(2)]
    means += [library.zmean_source(tomo) for tomo in range(2)]
    return np.concatenate([np.asarray(values).ravel(), np.asarray(means)])


@pytest.mark.parametrize("mode,factor", ((0, 3), (1, 3), (1, 7)))
@pytest.mark.parametrize("irregular", (False, True))
def test_zmid_convention_round_trip(irregular, mode, factor, tmp_path):
    """Z_LOW <-> Z_MID flips rebuild curves and means, and restore bitwise."""
    library = compiled_readers()
    filename = write_histogram(tmp_path, irregular)
    try:
        ci.initial_setup()
        ci.init_accuracy_boost(accuracy_boost=1, integration_accuracy=0)
        ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
        ci.init_photoz_histogram_akima(enabled=mode, preparation_factor=factor)
        load_sample(filename)
        base = state(library)

        # Repeating the same selection must not move a single bit.
        ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
        np.testing.assert_array_equal(state(library), base)

        # The flip reads the same z column as sample points instead of
        # left edges, which moves every histogram edge by half a cell:
        # the curves and the fiducial means must change together, on
        # the very next query.
        ci.init_photoz_conventions(interpolation_type=0, zmid_convention=1)
        flipped = state(library)
        assert np.all(np.isfinite(flipped))
        assert not np.array_equal(flipped[:-NMEANS], base[:-NMEANS])
        assert not np.array_equal(flipped[-NMEANS:], base[-NMEANS:])

        # Returning to the original reading restores the original state
        # exactly; no residue of the visited selection may survive.
        ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
        np.testing.assert_array_equal(state(library), base)
    finally:
        ci.initial_setup()


@pytest.mark.parametrize("first,second", (
    ((0, 3), (1, 3)),   # midpoint curve <-> Akima reconstruction
    ((1, 3), (1, 7)),   # Akima preparation factor
    ((1, 7), (0, 3)),   # Akima back to the midpoint curve
))
def test_reconstruction_round_trip(first, second, tmp_path):
    """Reconstruction and factor switches take effect and restore bitwise."""
    library = compiled_readers()
    filename = write_histogram(tmp_path, irregular=True)
    try:
        ci.initial_setup()
        ci.init_accuracy_boost(accuracy_boost=1, integration_accuracy=0)
        ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
        ci.init_photoz_histogram_akima(enabled=first[0],
                                       preparation_factor=first[1])
        load_sample(filename)
        base = state(library)

        ci.init_photoz_histogram_akima(enabled=second[0],
                                       preparation_factor=second[1])
        switched = state(library)
        assert np.all(np.isfinite(switched))
        assert not np.array_equal(switched, base)

        ci.init_photoz_histogram_akima(enabled=first[0],
                                       preparation_factor=first[1])
        np.testing.assert_array_equal(state(library), base)
    finally:
        ci.initial_setup()


def test_interpolant_round_trip(tmp_path):
    """The midpoint curve's stage-1 interpolant switch restores bitwise."""
    library = compiled_readers()
    filename = write_histogram(tmp_path, irregular=True)
    try:
        ci.initial_setup()
        ci.init_accuracy_boost(accuracy_boost=1, integration_accuracy=0)
        ci.init_photoz_histogram_akima(enabled=0, preparation_factor=3)
        ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
        load_sample(filename)
        base = state(library)

        # cspline (0) to linear (1) changes the midpoint reconstruction.
        ci.init_photoz_conventions(interpolation_type=1, zmid_convention=0)
        switched = state(library)
        assert np.all(np.isfinite(switched))
        assert not np.array_equal(switched, base)

        ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
        np.testing.assert_array_equal(state(library), base)
    finally:
        ci.initial_setup()
