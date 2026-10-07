"""Project choices for the shared galaxy/shear covariance notebook.

This module initializes 6 lens and 4 source distributions from the
project. Numerical algorithms live in cosmolike_notebook_utils.covariance.
The example is a massless-neutrino forecast with explicit Gaussian
non-Limber/IA choices and
number densities; it does not reproduce the project's frozen likelihood.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
    """Return resolved survey, cosmology and YAML accuracy choices.

    The covariance README records the catalog assumptions and their sources.
    Redshift-file normalization sets a shape, not a catalog number density.
    Arguments:
        accuracy_boost = None uses default.yaml; 1, 2, 4 or 8 refines it.
        gaussian = optional nonlimber/ia/A1/A2/B_TA model mapping.
        accuracy_overrides = named internal controls from default.yaml.
    Returns:
        Fully resolved settings, including the unboosted accuracy parameters.
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # Inclusive bands count every integer multipole once.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)

    # The fiducial is shared by G, SSC and cNG; CAMB runs only once.
    settings = {
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.05,
            "H0": 70.0,
            "ns": 0.965,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },

        # File columns describe radial shapes; these flags fix their z convention.
        "lens_file": "data/nz_maglim_Y3_unblinded_02_26_21.txt",
        "source_file": "data/nz_source_Y3_unblinded_02_26_21.txt",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,
        # Unit width factors keep the supplied lens n(z) shapes unchanged.
        "lens_photoz_stretch": [1.0]*6,

        # Measured pair and band choices are fixed during accuracy refinement.
        "excluded_gammat": [],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),

        # Densities are per square arcminute. Shape noise is per component.
        "area_deg2": 4143.0,
        "lens_density_arcmin2": [0.150, 0.107, 0.109, 0.146, 0.106, 0.100],
        "source_density_arcmin2": [1.475584985490327, 1.479383426887689,
                                    1.483671693529899, 1.461247850098986],
        "sigma_e_component": [0.2435002682964671, 0.2621945854120855,
                              0.2588927276307921, 0.3096827008528927],
        "bias": [1.5, 1.6, 1.7, 1.8, 2.0, 2.2],

        # Shell edges increase in a, from the distant boundary to the observer.
        "theta_edges_arcmin": np.geomspace(start=0.25, stop=250.0, num=31),
        "a_edges": 1.0/(1.0+np.array([3.1, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    Arguments:
        interface = imported cosmolike_desy1xplanck_interface module.
        settings = resolved mapping from configuration().
    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the galaxy/shear forecast with G, SSC, connected and total matrices.

    Arguments: interface = initialized compiled project; settings = configuration();
        space = "real" or "fourier"; rows = optional measured row subset;
        progress = optional (stage, elapsed_seconds) callback;
        backend = None for notebook wrappers, interface.covariance for CLI.
    Returns: shared forecast dict, including resolved settings and coordinates.
    The full real layout has 1500 entries; Fourier has 600 entries.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
