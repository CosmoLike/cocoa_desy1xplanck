"""Survey choices of the desy1xplanck galaxy/shear covariance forecast.

The covariance machinery is shared by every project and lives in
cosmolike_notebook_utils.covariance (external_modules/code/cosmolike_core).
This module supplies what belongs to this project: the n(z) files of the
6 MagLim lens bins and the 4 source bins, the catalog densities and shape
noise, the linear galaxy bias, the angular and multipole binning, and the
fiducial cosmology. The command-line runner compute_covariance.py and the
notebook EXAMPLE_EVALUATE_COVARIANCE.ipynb call its three functions in
order: configuration, initialize, compute.

The forecast covers the galaxy/shear sector of the data vector (cosmic
shear, galaxy-galaxy lensing and clustering: 1500 entries in real space,
600 in Fourier space); the CMB-lensing blocks are not computed. Its
physical choices are its own (massless neutrinos, linear galaxy bias, no
magnification, unit photo-z stretch, and the explicit Gaussian
non-Limber and intrinsic-alignment options of configuration), so it does
not reproduce the supplied covariance file that the likelihood reads.
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
    """Return the resolved survey, cosmology and accuracy settings.

    The survey numbers below describe the DES Y3 catalogs; the covariance
    README records the assumptions and their sources. Each bin of an n(z)
    file is normalized to unit integral, so the files set only the shape
    of each distribution; the number densities below set the noise.

    Arguments:
      accuracy_boost = None keeps the accuracy values of default.yaml (the
                       file next to this module); 1, 2, 4 or 8 refines them.
      gaussian = optional mapping of the Gaussian-only choices: nonlimber
                 (bool), ia (none, NLA or TATT) and the amplitudes A1, A2,
                 B_TA (gaussian_model of cosmolike_notebook_utils checks
                 them).
      accuracy_overrides = named internal accuracy settings of
                 default.yaml, given as keyword arguments (the ** collects
                 them into a dictionary).

    Returns:
      dict with every setting resolved, the accuracy parameters before the
      boost included.
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # Fourier space: 16 edges give 15 bands from l = 30 to 4000, spaced
    # logarithmically and rounded to integers (rint). A band [first, last]
    # includes both ends, so every integer multipole lies in one band.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)

    # The fiducial is shared by G, SSC and cNG; CAMB runs only once. H0 in
    # km/s/Mpc, As_1e9 = 10^9 A_s, w0pwa = w0 + wa, mnu in eV (0: massless
    # neutrinos), kmax in 1/Mpc (CAMB's largest wavenumber); non_linear_emul
    # 2 = CAMB's halofit, in its takahashi version.
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

        # The n(z) files, relative to the project folder. The two flags read
        # them as the likelihood does by default: cubic-spline interpolation
        # (0), and a z column that holds the left bin edges (0).
        "lens_file": "data/nz_maglim_Y3_unblinded_02_26_21.txt",
        "source_file": "data/nz_source_Y3_unblinded_02_26_21.txt",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,
        # The per-bin lens photo-z stretch (DES_DZ2_L<i> in the likelihood)
        # set to 1: the lens n(z) keep the shapes of the file.
        "lens_photoz_stretch": [1.0]*6,

        # excluded_gammat: (lens, source) pairs left out of gamma_t (none);
        # band_first, band_last: the Fourier bands above; lnm_edges: the
        # ln(M/(M_sun/h)) panel edges of the halo-mass integrals. These stay
        # fixed when the accuracy settings are refined.
        "excluded_gammat": [],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),

        # Survey area in square degrees (the DES Y3 footprint), number
        # densities in galaxies per square arcminute for each lens and
        # source bin, the shape dispersion per ellipticity component of each
        # source bin, and the linear galaxy bias of each lens bin (the
        # fiducial DES_B1_<i> of the likelihood).
        "area_deg2": 4143.0,
        "lens_density_arcmin2": [0.150, 0.107, 0.109, 0.146, 0.106, 0.100],
        "source_density_arcmin2": [1.475584985490327, 1.479383426887689,
                                    1.483671693529899, 1.461247850098986],
        "sigma_e_component": [0.2435002682964671, 0.2621945854120855,
                              0.2588927276307921, 0.3096827008528927],
        "bias": [1.5, 1.6, 1.7, 1.8, 2.0, 2.2],

        # theta_edges_arcmin: 31 edges, so 30 logarithmic angular bins from
        # 0.25 to 250 arcmin. a_edges: the edges of the line-of-sight
        # integration panels in scale factor a = 1/(1+z), increasing from
        # z = 3.1 (beyond the source n(z), which ends at z = 2.99) to
        # z = 1e-5, just short of the observer (a = 1); every panel gets its
        # own Gauss-Legendre nodes.
        "theta_edges_arcmin": np.geomspace(start=0.25, stop=250.0, num=31),
        "a_edges": 1.0/(1.0+np.array([3.1, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the forecast state in the interface.

    No covariance is computed here: initialize_forecast hands the CAMB
    power spectra, growth and distances, the n(z) files and the forecast
    nuisance values to the compiled interface.

    Arguments:
      interface = the imported cosmolike_desy1xplanck_interface module.
      settings = the resolved mapping of configuration().

    Returns:
      dict of the installed CAMB tables (in the set_cosmology format),
      suitable for saving next to the results.

    Side effects:
      replaces the global cosmology and nuisance state of the interface.
      The likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the galaxy/shear forecast with its G, SSC, cNG and total parts.

    Arguments:
      interface = the interface after initialize().
      settings = the resolved mapping of configuration().
      space = "real" (angular bins, 1500 entries) or "fourier" (bandpowers,
              600 entries: one E-mode shear spectrum replaces xi_+ and xi_-).
      rows = None for the full layout, or a subset of the measured rows.
      progress = None, or a function called with (stage, elapsed_seconds).
      backend = None for the notebook wrappers, interface.covariance for
                the production bindings of the command-line runner.

    Returns:
      the shared forecast dict: the matrices, the resolved settings and the
      coordinates of the rows.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
