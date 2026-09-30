"""Notebook wrappers for the desy1xplanck Cosmolike interface.

The EXAMPLE_EVALUATE notebooks all drive the same compiled interface
(cosmolike_desy1xplanck_interface) through the same steps: run CAMB
once (cnu.get_camb_cosmology), push the resulting power spectra and
distances into the interface with set_cosmology, set the nuisance
parameters, and read off a spectrum, a correlation function, or the
masked chi2. This module holds those steps once, so every notebook of
this project imports the same wrappers instead of redefining them:

    import cosmolike_desy1xplanck_notebook_wrappers as nw
    nw.configure(lmax=75000)          # this notebook's yaml values
    nw.init_cosmolike(CLprobe="xi", with_data=True)
    nw.get_chi2(omegam=0.31)

Three layers of state matter here:

- The compiled interface is a C library with global state: every
  init_* and set_* call above replaces part of it, and the spectrum
  functions read whatever was set last. Each wrapper therefore
  resets everything it depends on (tables, accuracy, cosmology,
  nuisances) on every call, so no call depends on which wrapper ran
  before it.
- The project fiducial point lives in this module as plain
  constants (DES_A1_1, ...), shared by every notebook; a notebook
  overrides any of them per call (nw.C_ss_tomo_limber(ell=ell,
  omegam=x)) or imports the names for its own sweeps.
- The few values that differ between notebooks because each mirrors
  its own yaml (the lmax of the internal C_ell tables, the angular
  binning, the nonlinear emulator choice) live in _CONFIG and are
  set once per notebook with configure().

Every wrapper accepts the same accuracy arguments and applies the
same folds: CLAccuracyBoost multiplies by AccuracyBoost, the
integration accuracy grows as |3 (CLAccuracyBoost - 1)|, and the
C_ell table reaches lmax + 20000 (CLAccuracyBoost - 1).

Two desy1xplanck-specific points, both mirroring the likelihood
(_cosmolike_prototype_base.py):

- init_cosmolike also initializes the CMB-lensing machinery (the
  beam/pixel-window filter of the kappa cross-correlations and the
  kk bandpower binning), read from the same .dataset keys the
  likelihood reads, so the ks/gk/kk probes work from notebooks.
- The lens (MagLim) photo-z model has a stretch parameter per bin
  (DES_DZ2_L*) on top of the additive bias, and the magnification
  coefficients DES_BMAG_* are fixed nonzero values.
"""

import os
import sys

import numpy as np
from getdist import IniFile

# the shared notebook utilities live in cosmolike_core; the compiled
# interface is on the path already (each project's interface/
# directory is part of the Cocoa PYTHONPATH)
sys.path.insert(0, os.environ["ROOTDIR"] + "/external_modules/code/cosmolike_core")
import cosmolike_notebook_utils as cnu
import cosmolike_desy1xplanck_interface as ci


# ----------------------------------------------------------------------
# Project fiducial point (the evaluate override of the example yamls;
# the lens values are the params_lens_maglim.yaml reference point)
# ----------------------------------------------------------------------
As_1e9 = 2.1
ns = 0.96605
H0 = 67.32
omegab = 0.04
omegam = 0.3
mnu = 0.06
w = -1.0
w0pwa = -1.0
DES_A1_1 = -0.7      # NLA amplitude
DES_A1_2 = -1.7      # NLA redshift power-law index
DES_DZ_S1 = 0.0
DES_DZ_S2 = 0.0
DES_DZ_S3 = 0.0
DES_DZ_S4 = 0.0
DES_M1 = -0.0063
DES_M2 = -0.0198
DES_M3 = -0.0241
DES_M4 = -0.0369
DES_DZ_L1 = -0.009
DES_DZ_L2 = -0.035
DES_DZ_L3 = -0.005
DES_DZ_L4 = -0.007
DES_DZ_L5 = 0.002
DES_DZ_L6 = 0.002
DES_DZ2_L1 = 0.975   # MagLim lens photo-z stretch
DES_DZ2_L2 = 1.306
DES_DZ2_L3 = 0.87
DES_DZ2_L4 = 0.918
DES_DZ2_L5 = 1.08
DES_DZ2_L6 = 0.845
DES_B1_1 = 1.5
DES_B1_2 = 1.6
DES_B1_3 = 1.7
DES_B1_4 = 1.8
DES_B1_5 = 2.0
DES_B1_6 = 2.2
DES_BMAG_1 = 0.43    # MagLim magnification coefficients (fixed)
DES_BMAG_2 = 0.30
DES_BMAG_3 = 1.75
DES_BMAG_4 = 1.94
DES_BMAG_5 = 1.56
DES_BMAG_6 = 2.96
DES_PM1 = 0.0
DES_PM2 = 0.0
DES_PM3 = 0.0
DES_PM4 = 0.0
DES_PM5 = 0.0
DES_PM6 = 0.0

# default nuisance vectors built from the constants above; wrappers
# take None and fall back to these, so a call overrides one vector
# without retyping the rest
A1_FID = [DES_A1_1, DES_A1_2, 0, 0]
A2_FID = [0, 0, 0, 0]
BTA_FID = [0, 0, 0, 0]
SHEAR_PHOTOZ_FID = [DES_DZ_S1, DES_DZ_S2, DES_DZ_S3, DES_DZ_S4]
M_FID = [DES_M1, DES_M2, DES_M3, DES_M4]
LENS_PHOTOZ_FID = [DES_DZ_L1, DES_DZ_L2, DES_DZ_L3,
                   DES_DZ_L4, DES_DZ_L5, DES_DZ_L6]
LENS_STRETCH_FID = [DES_DZ2_L1, DES_DZ2_L2, DES_DZ2_L3,
                    DES_DZ2_L4, DES_DZ2_L5, DES_DZ2_L6]
B1_FID = [DES_B1_1, DES_B1_2, DES_B1_3, DES_B1_4, DES_B1_5, DES_B1_6]
BMAG_FID = [DES_BMAG_1, DES_BMAG_2, DES_BMAG_3,
            DES_BMAG_4, DES_BMAG_5, DES_BMAG_6]
ZEROS6 = [0, 0, 0, 0, 0, 0]
PM_FID = [DES_PM1, DES_PM2, DES_PM3, DES_PM4, DES_PM5, DES_PM6]

# ----------------------------------------------------------------------
# Per-notebook configuration
# ----------------------------------------------------------------------
# Every notebook mirrors its own yaml, and these are the values that
# differ between them: EXAMPLE_EVALUATE1 (cosmic shear) uses
# lmax = 75000. The remaining entries are shared by the current
# notebooks and sit here so a future notebook can change them the
# same way.
_CONFIG = {
    "lmax": 75000,              # base of the internal C_ell tables
    "ntheta": 30,               # angular bins of the real-space vector
    "theta_min_arcmin": 0.25,
    "theta_max_arcmin": 250.0,
    "non_linear_emul": 2,       # 1 = EuclidEmulator2, 2 = halofit
    "path": "../../external_modules/data/desy1xplanck",
    "data_file": "Y3xPlanckPR4.dataset",
    "IA_model": 0,
    "IA_redshift_evolution": 3,
    "IA_code": 0,               # 0 = C FASTPT (NLA always uses 0)
    "bias_model": [0, 0, 0, 1, 0, 0],    # n(z) photo-z conventions (mirror the likelihood yaml keys):
    # interpolation 0 = cspline, 1 = linear, 2+ = Steffen monotone;
    # z column 0 = Z_LOW (left bin edges), 1 = Z_MID (sample points)
    "photoz_interpolation_type": 0,
    "photoz_zmid_convention": 0,
    # C-FAST-PT internal (convolution) grid / output grid; 1.0 = equal
    "internal_accuracyboost": 1.0,

}

# filled by init_cosmolike: the HDF5 file with every hydro simulation
# used by init_baryons_contamination
allsims = None


def configure(**overrides):
    """Sets this notebook's yaml-mirroring values, once per notebook.

    Every keyword must already exist in _CONFIG; an unknown name is
    almost always a typo, so it raises instead of being stored
    silently.

    Arguments:
      overrides = keyword form of any _CONFIG entry, e.g.
                  configure(lmax=75000).

    Returns:
      nothing; later wrapper calls read the stored values.

    Raises:
      KeyError naming the unknown keyword and the valid names.
    """
    for name, value in overrides.items():
        if name not in _CONFIG:
            raise KeyError(
                f"configure() got unknown option '{name}'; valid options: "
                + ", ".join(sorted(_CONFIG)))
        _CONFIG[name] = value


def init_cosmolike(CLprobe=None, with_data=False, lmax=None):
    """One-time interface setup for a notebook session.

    Reads the project's .dataset file (the small text file listing
    the n(z), covariance, mask, data-vector, and CMB-lensing files),
    then runs the interface init sequence the likelihood
    (_cosmolike_prototype_base.initialize) runs: the angular binning,
    the photo-z conventions, the CMB-lensing beam/pixel filter and kk
    bandpower binning, the n(z) tables, and the IA model. The chi2
    machinery (probes, covariance, mask, data vector) only loads when
    asked, because the plotting-only notebooks never need it.

    Arguments:
      CLprobe   = "xi", "6x2pt", ... to select the probe set and
                  (except for the shear-only probes) the galaxy-bias
                  model, or None to skip init_probes for a
                  plotting-only session.
      with_data = True also loads covariance, mask, and data vector,
                  which get_chi2 needs.
      lmax      = base lmax of the internal C_ell tables, or None
                  for the configure()d value.

    Returns:
      the parsed IniFile, for notebooks that read extra entries.

    Side effects:
      replaces the compiled interface's global state and sets this
      module's allsims to the hydro-simulation file of the dataset.
    """
    global allsims
    if lmax is None:
        lmax = _CONFIG["lmax"]
    ini = IniFile(os.path.normpath(
        os.path.join(_CONFIG["path"], _CONFIG["data_file"])))
    allsims = ini.relativeFileName('all_sims_hdf5_file')
    ci.initial_setup()
    if CLprobe is not None:
        ci.init_probes(possible_probes=CLprobe)
    ci.init_binning(int(ini.int("n_theta")),
                    ini.float("theta_min_arcmin"),
                    ini.float("theta_max_arcmin"))
    ci.init_photoz_conventions(
        int(_CONFIG["photoz_interpolation_type"]),
        int(_CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(
        float(_CONFIG["internal_accuracyboost"]))
    # CMB-lensing cross-correlation (beam + healpix pixel window) and
    # kk auto bandpowers, exactly as the likelihood initializes them
    ci.init_cmb_cross_correlation(
        lmin=ini.int("lmin_kx"),
        lmax=ini.int("lmax_kx"),
        fwhm=ini.float("fwhm_kx"),
        healpixwin_filename=ini.relativeFileName('healpix_win_func_kx_file'))
    nbins_kk = ini.int("nbp_kk")
    nvar_kk = ini.float("hartlap_nvar_kk")
    ci.init_cmb_auto_bandpower(
        nbins=nbins_kk,
        lmin=ini.int("lminbp_kk"),
        lmax=ini.int("lmaxbp_kk"),
        binning_matrix=ini.relativeFileName('binmat_kk_file'),
        theory_offset=ini.relativeFileName('offset_kk_file'),
        alpha=(nvar_kk - nbins_kk - 2.0)/(nvar_kk - 1.0))
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_IA(ia_model=int(_CONFIG["IA_model"]),
               ia_redshift_evolution=int(_CONFIG["IA_redshift_evolution"]),
               ia_code=int(_CONFIG["IA_code"]))
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=ini.relativeFileName('nz_lens_file'),
        lens_ntomo=int(ini.int("lens_ntomo")),
        source_multihisto_file=ini.relativeFileName('nz_source_file'),
        source_ntomo=int(ini.int("source_ntomo")))
    if with_data:
        ci.init_data_real(ini.relativeFileName('cov_file'),
                          ini.relativeFileName('mask_file'),
                          ini.relativeFileName('data_file'))
    if CLprobe is not None and CLprobe not in ("xi",
                                               "2x2pt_ss_sk",
                                               "3x2pt_ss_sk_sk"):
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    ci.init_ntable_lmax(lmax=int(lmax))
    ci.init_accuracy_boost(1.0, int(1))
    return ini


def _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=None, M=None, shear_photoz_bias=None,
               A1=None, A2=None, BTA=None,
               lens_photoz_bias=None, lens_photoz_stretch=None,
               B1=None, B2=None,
               B_MAG=None, B3nl=None, BK=None, PM=None,
               baryon_sims=None, allsims_file=None):
    """Runs CAMB and pushes one complete state into the interface.

    This is the body every wrapper shares. The compiled interface
    keeps global state, so the sequence rebuilds everything a
    spectrum call reads: the accuracy folds and lookup tables, the
    binning when a real-space probe asked for it, the cosmology
    (power spectra, growth, distances from one CAMB run), and each
    nuisance group whose vectors were passed. A group passed as None
    is skipped, which leaves that part of the state at whatever the
    interface holds, exactly as the per-probe notebook definitions
    did (a cosmic-shear wrapper never touched galaxy bias).

    Arguments:
      omegam ... non_linear_emul = the cosmology and accuracy
                 arguments, forwarded to cnu.get_camb_cosmology
                 (kmax in h/Mpc; see its docstring for the grids).
      binning  = (ntheta, theta_min_arcmin, theta_max_arcmin) to
                 re-run init_binning (the real-space wrappers), or
                 None to keep the current binning.
      M, shear_photoz_bias, A1, A2, BTA = shear nuisance vectors
                 (M and the photo-z shifts gate the shear setters;
                 A1 gates the IA setter).
      lens_photoz_bias, lens_photoz_stretch, B1, B2, B_MAG, B3nl,
      BK       = clustering nuisance vectors (the MagLim photo-z
                 model has a stretch per bin on top of the additive
                 bias); passing B1 also selects the configured
                 galaxy-bias model.
      PM       = point-mass amplitudes, one per lens bin, or None.
      baryon_sims = a hydro simulation name to contaminate the
                 matter power with, or None to reset that state.
      allsims_file = HDF5 file for baryon_sims, or None for the one
                 init_cosmolike recorded.

    Returns:
      nothing; the interface state is the result.
    """
    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_growth, z_interp_1D, chi) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, AccuracyBoost=AccuracyBoost,
        kmax=kmax, k_per_logint=k_per_logint,
        CAMBAccuracyBoost=CAMBAccuracyBoost,
        CLAccuracyBoost=CLAccuracyBoost,
        non_linear_emul=non_linear_emul)
    # the house accuracy folds: the overall boost multiplies the
    # cosmolike boost, and the integration accuracy and the C_ell
    # table length grow with it
    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(
        0, CLIntegrationAccuracy + abs(3*(CLAccuracyBoost - 1.0)))
    ci.init_ntable_lmax(int(_CONFIG["lmax"] + 20000*(CLAccuracyBoost - 1)))
    ci.init_accuracy_boost(CLAccuracyBoost, int(CLIntegrationAccuracy))
    ci.init_photoz_conventions(
        int(_CONFIG["photoz_interpolation_type"]),
        int(_CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(
        float(_CONFIG["internal_accuracyboost"]))
    if binning is not None:
        ci.init_binning(int(binning[0]), binning[1], binning[2])
    if B1 is not None:
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    # the growth table has its own z grid (z_growth, the dense 1D grid
    # cut at the last z_2D node), handed over as z_G, as the likelihood
    # does
    ci.set_cosmology(omegam=omegam,
                     H0=H0,
                     log10k_2D=log10k_interp_2D,
                     z_2D=z_interp_2D,
                     lnP_linear=lnPL,
                     lnP_nonlinear=lnPNL,
                     G=G_growth,
                     z_G=z_growth,
                     z_1D=z_interp_1D,
                     chi=chi)
    if M is not None:
        ci.set_nuisance_shear_calib(M=M)
    if shear_photoz_bias is not None:
        ci.set_nuisance_shear_photoz(bias=shear_photoz_bias)
    if lens_photoz_bias is not None:
        ci.set_nuisance_clustering_photoz(bias=lens_photoz_bias,
                                          stretch=lens_photoz_stretch)
    if B1 is not None:
        ci.set_nuisance_bias(B1=B1, B2=B2, B_MAG=B_MAG, B3nl=B3nl, BK=BK)
    if A1 is not None:
        ci.set_nuisance_ia(A1=A1, A2=A2, B_TA=BTA)
    if PM is not None:
        ci.set_point_mass(PMV=PM)
    if baryon_sims is None:
        ci.reset_bary_struct()
    else:
        if allsims_file is None:
            allsims_file = allsims
        ci.init_baryons_contamination(sim=baryon_sims, allsims=allsims_file)


def _shear_defaults(M, shear_photoz_bias, A1, A2, BTA):
    """Replaces None shear vectors with the fiducial ones."""
    if M is None:
        M = M_FID
    if shear_photoz_bias is None:
        shear_photoz_bias = SHEAR_PHOTOZ_FID
    if A1 is None:
        A1 = A1_FID
    if A2 is None:
        A2 = A2_FID
    if BTA is None:
        BTA = BTA_FID
    return M, shear_photoz_bias, A1, A2, BTA


def _clustering_defaults(lens_photoz_bias, lens_photoz_stretch,
                         B1, B2, B_MAG, B3nl, BK):
    """Replaces None clustering vectors with the fiducial ones."""
    if lens_photoz_bias is None:
        lens_photoz_bias = LENS_PHOTOZ_FID
    if lens_photoz_stretch is None:
        lens_photoz_stretch = LENS_STRETCH_FID
    if B1 is None:
        B1 = B1_FID
    if B2 is None:
        B2 = ZEROS6
    if B_MAG is None:
        B_MAG = BMAG_FID
    if B3nl is None:
        B3nl = ZEROS6
    if BK is None:
        BK = ZEROS6
    return (lens_photoz_bias, lens_photoz_stretch,
            B1, B2, B_MAG, B3nl, BK)


def C_ss_tomo_limber(ell, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                     As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                     A1=None, A2=None, BTA=None,
                     shear_photoz_bias=None, M=None,
                     baryon_sims=None, AccuracyBoost=1.0, kmax=10.0,
                     k_per_logint=10, CAMBAccuracyBoost=1.0,
                     CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                     non_linear_emul=None, allsims=None):
    """Cosmic-shear angular power spectra (EE, BB) at multipoles ell.

    Rebuilds the full interface state (see _set_state) and evaluates
    ci.C_ss_tomo_limber. The nuisance vectors default to the module
    fiducials when passed as None.

    Arguments:
      ell = 1D array of multipoles; the rest as in _set_state, with
      the shear group only (this wrapper never touches clustering).

    Returns:
      (EE, BB): two 3D arrays (n_ell, n_bin, n_bin).
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.C_ss_tomo_limber(l=ell)


def xi(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
       omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
       w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
       shear_photoz_bias=None, M=None, baryon_sims=None,
       AccuracyBoost=1.0, kmax=10.0, k_per_logint=10,
       CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
       CLIntegrationAccuracy=0, non_linear_emul=None, allsims=None):
    """Real-space shear correlations xi_plus/minus on a theta grid.

    Same state build as C_ss_tomo_limber plus a re-binning, so the
    binning can change between calls without restarting the kernel;
    the binning arguments default to the configure()d values.

    Returns:
      (theta, xi_plus, xi_minus): theta in arcmin, xi 3D arrays
      (n_theta, n_bin, n_bin).
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    (xip, xim) = ci.xi_pm_tomo()
    return (ci.get_binning_real_space(), xip, xim)


def get_chi2(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
             w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
             shear_photoz_bias=None, M=None,
             lens_photoz_bias=None, lens_photoz_stretch=None,
             galaxy_bias_b1=None,
             galaxy_bias_b2=None, galaxy_bias_bmag=None,
             galaxy_bias_b3nl=None, galaxy_bias_bk=None, PM=None,
             baryon_sims=None, AccuracyBoost=1.0, kmax=7.5,
             k_per_logint=10, CAMBAccuracyBoost=1.0,
             CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
             non_linear_emul=None, allsims=None):
    """chi2 of the masked data vector against the loaded data.

    Requires init_cosmolike(CLprobe=..., with_data=True) first: the
    probe selection fixes which blocks enter the masked vector, and
    with_data loads the covariance, mask, and data vector this chi2
    compares against. The full 6x2pt nuisance state is set every
    call; blocks outside the selected probes simply never read
    theirs (a "xi" run ignores the clustering state).

    Returns:
      float chi2.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    if PM is None:
        PM = PM_FID
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, lens_photoz_stretch, galaxy_bias_b1,
     galaxy_bias_b2, galaxy_bias_bmag, galaxy_bias_b3nl,
     galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, lens_photoz_stretch, galaxy_bias_b1,
        galaxy_bias_b2, galaxy_bias_bmag, galaxy_bias_b3nl,
        galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias,
               lens_photoz_stretch=lens_photoz_stretch,
               B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk, PM=PM,
               baryon_sims=baryon_sims, allsims_file=allsims)
    datavector = np.array(ci.compute_data_vector_masked())
    return ci.compute_chi2(datavector)


# ----------------------------------------------------------------------
# Response functions (cosmic shear)
# ----------------------------------------------------------------------
def dlnC_dlss_tomo_limber(k, ell, omegam=omegam, omegab=omegab, H0=H0,
                          ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                          A1=None, A2=None, BTA=None,
                          shear_photoz_bias=None, M=None,
                          baryon_sims=None, AccuracyBoost=1.0,
                          kmax=10.0, k_per_logint=10,
                          CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                          CLIntegrationAccuracy=0,
                          non_linear_emul=None, allsims=None):
    """Response d ln C_ss / d ln k at wavenumbers k, multipoles ell.

    Shear state as in C_ss_tomo_limber, then the interface's
    response evaluation.

    Returns:
      array as ci.dlnC_ss_dlnk_tomo_limber returns it.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.dlnC_ss_dlnk_tomo_limber(k=k, l=ell)


def dlnxi_dlnk_pm_tomo_limber(k, ntheta=None, theta_min_arcmin=None,
                              theta_max_arcmin=None, omegam=omegam,
                              omegab=omegab, H0=H0, ns=ns,
                              As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                              A1=None, A2=None, BTA=None,
                              shear_photoz_bias=None, M=None,
                              baryon_sims=None, AccuracyBoost=1.0,
                              kmax=10.0, k_per_logint=10,
                              CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                              CLIntegrationAccuracy=0,
                              non_linear_emul=None, allsims=None):
    """Response d ln xi_pm / d ln k at wavenumbers k.

    Shear state plus a re-binning, as in xi.

    Returns:
      (theta, dlnxip_dlnk, dlnxim_dlnk).
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    (dlnxip_dlnk, dlnxim_dlnk) = ci.dlnxi_dlnk_pm_tomo_limber(k=k)
    return (ci.get_binning_real_space(), dlnxip_dlnk, dlnxim_dlnk)


def rf_C_ss_tomo_limber(k, ell, omegam=omegam, omegab=omegab, H0=H0,
                        ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                        A1=None, A2=None, BTA=None,
                        shear_photoz_bias=None, M=None,
                        baryon_sims=None, AccuracyBoost=1.0,
                        kmax=10.0, k_per_logint=10,
                        CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                        CLIntegrationAccuracy=0,
                        non_linear_emul=None, allsims=None):
    """Cumulative response R(k_max) of C_ss.

    Shear state as in C_ss_tomo_limber, then ci.rf_C_ss_tomo_limber.

    Returns:
      array as ci.rf_C_ss_tomo_limber returns it.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.rf_C_ss_tomo_limber(k=k, l=ell)


def rf_xi_tomo_limber(k, ntheta=None, theta_min_arcmin=None,
                      theta_max_arcmin=None, omegam=omegam,
                      omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
                      w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
                      shear_photoz_bias=None, M=None,
                      baryon_sims=None, AccuracyBoost=1.0, kmax=10.0,
                      k_per_logint=10, CAMBAccuracyBoost=1.0,
                      CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                      non_linear_emul=None, allsims=None):
    """Cumulative response R(k_max) of xi_pm.

    Shear state plus a re-binning, then ci.rf_xi_tomo_limber.

    Returns:
      (theta, rf_xip, rf_xim).
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    (rf_xip, rf_xim) = ci.rf_xi_tomo_limber(k=k)
    return (ci.get_binning_real_space(), rf_xip, rf_xim)


# ----------------------------------------------------------------------
# CMB lensing x shear (the CMB filter and bandpower machinery is set up
# by init_cosmolike, mirroring the likelihood)
# ----------------------------------------------------------------------
def C_ks_tomo_limber(ell, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                     As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                     A1=None, A2=None, BTA=None,
                     shear_photoz_bias=None, M=None,
                     baryon_sims=None, AccuracyBoost=1.0, kmax=10.0,
                     k_per_logint=10, CAMBAccuracyBoost=1.0,
                     CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                     non_linear_emul=None, allsims=None):
    """CMB lensing x shear angular power spectra at multipoles ell.

    Shear state as in C_ss_tomo_limber (the CMB is a single lens
    plane, so only the shear nuisances enter), then
    ci.C_ks_tomo_limber.

    Returns:
      2D array (n_ell, n_bin).
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.C_ks_tomo_limber(l=ell)


def w_ks(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
         omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
         w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
         shear_photoz_bias=None, M=None, baryon_sims=None,
         AccuracyBoost=1.0, kmax=10.0, k_per_logint=10,
         CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
         CLIntegrationAccuracy=0, non_linear_emul=None, allsims=None):
    """Real-space CMB lensing x shear w_ks(theta) on a theta grid.

    Same state build as C_ks_tomo_limber plus a re-binning; the CMB
    beam/pixel-window filter set by init_cosmolike enters the
    projection inside ci.w_ks_tomo.

    Returns:
      (theta, w_ks): theta in arcmin, w_ks a 2D array
      (n_theta, n_bin).
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return (ci.get_binning_real_space(), ci.w_ks_tomo())


def dlnC_ks_dlnk_tomo_limber(k, ell, omegam=omegam, omegab=omegab,
                             H0=H0, ns=ns, As_1e9=As_1e9, w=w,
                             w0pwa=w0pwa, A1=None, A2=None, BTA=None,
                             shear_photoz_bias=None, M=None,
                             baryon_sims=None, AccuracyBoost=1.0,
                             kmax=10.0, k_per_logint=10,
                             CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                             CLIntegrationAccuracy=0,
                             non_linear_emul=None, allsims=None):
    """Response d ln C_ks / d ln k at wavenumbers k, multipoles ell.

    Shear state as in C_ks_tomo_limber, then the interface's
    response evaluation.

    Returns:
      array as ci.dlnC_ks_dlnk_tomo_limber returns it.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.dlnC_ks_dlnk_tomo_limber(k=k, l=ell)


def rf_C_ks_tomo_limber(k, ell, omegam=omegam, omegab=omegab, H0=H0,
                        ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                        A1=None, A2=None, BTA=None,
                        shear_photoz_bias=None, M=None,
                        baryon_sims=None, AccuracyBoost=1.0,
                        kmax=10.0, k_per_logint=10,
                        CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                        CLIntegrationAccuracy=0,
                        non_linear_emul=None, allsims=None):
    """Cumulative response R(k_max) of C_ks.

    Shear state as in C_ks_tomo_limber, then ci.rf_C_ks_tomo_limber.

    Returns:
      array as ci.rf_C_ks_tomo_limber returns it.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.rf_C_ks_tomo_limber(k=k, l=ell)


def dlnw_ks_dlnk_tomo(k, ntheta=None, theta_min_arcmin=None,
                      theta_max_arcmin=None, omegam=omegam,
                      omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
                      w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
                      shear_photoz_bias=None, M=None,
                      baryon_sims=None, AccuracyBoost=1.0, kmax=10.0,
                      k_per_logint=10, CAMBAccuracyBoost=1.0,
                      CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                      non_linear_emul=None, allsims=None):
    """Response d ln w_ks / d ln k at wavenumbers k.

    Shear state plus a re-binning, as in w_ks.

    Returns:
      (theta, dlnwks_dlnk).
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return (ci.get_binning_real_space(), ci.dlnw_ks_dlnk_tomo(k=k))


def rf_w_ks_tomo(k, ntheta=None, theta_min_arcmin=None,
                 theta_max_arcmin=None, omegam=omegam,
                 omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
                 w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
                 shear_photoz_bias=None, M=None,
                 baryon_sims=None, AccuracyBoost=1.0, kmax=10.0,
                 k_per_logint=10, CAMBAccuracyBoost=1.0,
                 CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                 non_linear_emul=None, allsims=None):
    """Cumulative response R(k_max) of w_ks.

    Shear state plus a re-binning, then ci.rf_w_ks_tomo.

    Returns:
      (theta, rf_wks).
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return (ci.get_binning_real_space(), ci.rf_w_ks_tomo(k=k))
