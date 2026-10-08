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
- The project fiducial point (the reference parameter values at
  which the examples evaluate the model) lives in this module as
  plain constants (DES_A1_1, ...), shared by every notebook; a
  notebook overrides any of them per call (nw.C_ss_tomo_limber(
  ell=ell, omegam=x)) or imports the names for its own sweeps.
- The few values that differ between notebooks because each mirrors
  its own yaml (the lmax of the internal C_ell tables, the angular
  binning, the nonlinear emulator choice) live in _CONFIG and are
  set once per notebook with configure().

Every wrapper accepts the same accuracy arguments and combines them
by the same rules: CLAccuracyBoost is multiplied by AccuracyBoost,
the integration accuracy grows by |3 (CLAccuracyBoost - 1)|, and the
C_ell tables reach lmax + 20000 (CLAccuracyBoost - 1).

Two desy1xplanck-specific points, both mirroring the likelihood
(_cosmolike_prototype_base.py):

- init_cosmolike also initializes the CMB-lensing machinery (the
  beam/pixel-window filter of the kappa cross-correlations and the
  kk bandpower binning), read from the same .dataset keys the
  likelihood reads, so the ks/gk/kk probes work from notebooks.
- The lens (MagLim) photo-z model has, per bin i, a stretch s_i
  (DES_DZ2_L<i>) on top of the shift dz_i (DES_DZ_L<i>): cosmolike
  maps n_i(z) -> n_i((z - dz_i - zbar_i)/s_i + zbar_i)/s_i, with
  zbar_i the mean redshift of the tabulated bin, so s_i = 1 leaves
  the width unchanged. The magnification coefficients DES_BMAG_* are
  fixed nonzero values.
"""

import os
import sys

import numpy as np
from getdist import IniFile

# the shared notebook utilities live in cosmolike_core; the compiled
# interface is on the path already (each project's interface/
# directory is part of the Cocoa PYTHONPATH). ROOTDIR, the path of the
# Cocoa/ folder, is exported by start_cocoa.sh; without it this line
# stops with a KeyError.
sys.path.insert(0, os.environ["ROOTDIR"] + "/external_modules/code/cosmolike_core")
import cosmolike_notebook_utils as cnu
import cosmolike_desy1xplanck_interface as ci


# ----------------------------------------------------------------------
# Project fiducial point (the evaluate override of the example yamls;
# the lens values are the params_lens_maglim.yaml reference point)
# ----------------------------------------------------------------------
# Cosmology: As_1e9 = 10^9 A_s (primordial amplitude), ns (scalar
# spectral index), H0 in km/s/Mpc, omegab and omegam (density
# parameters today), mnu (sum of the neutrino masses, eV), w = w0 and
# w0pwa = w0 + wa of the dark-energy equation of state
# w(a) = w0 + wa (1 - a).
As_1e9 = 2.1
ns = 0.96605
H0 = 67.32
omegab = 0.04
omegam = 0.3
mnu = 0.06
w = -1.0
w0pwa = -1.0
# Intrinsic alignment, IA_redshift_evolution = 3: A1(z) = DES_A1_1
# ((1 + z)/(1 + z_0))^DES_A1_2, with z_0 a fixed pivot redshift
DES_A1_1 = -0.7      # NLA amplitude
DES_A1_2 = -1.7      # NLA redshift power-law index
# Source bins: photo-z shifts (redshift units) and multiplicative shear
# calibrations m_i (the shear of bin i is multiplied by 1 + m_i)
DES_DZ_S1 = 0.0
DES_DZ_S2 = 0.0
DES_DZ_S3 = 0.0
DES_DZ_S4 = 0.0
DES_M1 = -0.0063
DES_M2 = -0.0198
DES_M3 = -0.0241
DES_M4 = -0.0369
# Lens bins: photo-z shifts dz_i and stretches s_i (module docstring)
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
# Linear galaxy bias b1 of each lens bin
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
# Point masses of the lens bins (10^13 M_sun/h): a 1/theta^2 term in
# gamma_t; zero at the fiducial point
DES_PM1 = 0.0
DES_PM2 = 0.0
DES_PM3 = 0.0
DES_PM4 = 0.0
DES_PM5 = 0.0
DES_PM6 = 0.0

# default nuisance vectors built from the constants above; wrappers
# take None and fall back to these, so a call overrides one vector
# without retyping the rest. The IA vectors have one slot per source
# bin, but with IA_redshift_evolution = 3 cosmolike reads only the
# first two (amplitude, power-law index); ZEROS6 fills the lens-bin
# vectors that are zero at the fiducial point (B2, B3nl, BK).
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
    # edges of the logarithmic angular binning, in arcmin
    "theta_min_arcmin": 0.25,
    "theta_max_arcmin": 250.0,
    "non_linear_emul": 2,       # 1 = EuclidEmulator2, 2 = halofit
    # the folder of the .dataset file, relative to the folder the notebook
    # runs in (projects/desy1xplanck), and the .dataset file itself
    "path": "../../external_modules/data/desy1xplanck",
    "data_file": "Y3xPlanckPR4.dataset",
    # 0 = NLA intrinsic alignments, 1 = TATT; 3 = the redshift power law
    # of the IA amplitudes (see A1_FID)
    "IA_model": 0,
    "IA_redshift_evolution": 3,
    "IA_code": 0,               # 0 = C FASTPT (NLA always uses 0)
    # galaxy-bias model codes for (b1, b2, bs2, b3, bmag, bK): 0 = one
    # amplitude per lens bin, 1 = derived from b1 (here b3), as bias_model
    # in the likelihood yaml files
    "bias_model": [0, 0, 0, 1, 0, 0],    # n(z) photo-z conventions (mirror the likelihood yaml keys):
    # n(z) photo-z conventions, the likelihood yaml keys of the same names:
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
    # set before the init_accuracy_boost at the end of this function, as
    # in the likelihood: that is normally the first call of the process,
    # which stores the C-FAST-PT internal grid fraction
    # (internal_accuracyboost) it finds as the base every later call
    # multiplies by the boost; called the other way round, the base would
    # be the C default 0.5 that initial_setup restores
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
    spectrum call reads: the accuracy settings and lookup tables, the
    binning when a real-space probe asked for it, the cosmology
    (power spectra, growth, distances from one CAMB run), and each
    nuisance group whose vectors were passed. A group passed as None
    is skipped, which leaves that part of the state at whatever the
    interface holds: a cosmic-shear wrapper never touches the galaxy
    bias.

    The neutrino mass comes from the module constant mnu (0.06 eV),
    not from an argument. set_cosmology receives no cold dark matter +
    baryon spectrum here (the likelihood passes one as lnP_linear_cb),
    so after a wrapper call any halo-model quantity built on that
    spectrum (the halo statistics, or ci.sigma2 with field=1) makes
    cosmolike print a fatal error and end the Python process.

    Arguments:
      omegam ... non_linear_emul = the cosmology and accuracy
                 arguments, forwarded to cnu.get_camb_cosmology
                 (kmax in 1/Mpc; see its docstring for the grids).
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
     G_growth, z_growth, z_interp_1D, chi,
     omegan2, lnPL_cb) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, AccuracyBoost=AccuracyBoost,
        kmax=kmax, k_per_logint=k_per_logint,
        CAMBAccuracyBoost=CAMBAccuracyBoost,
        CLAccuracyBoost=CLAccuracyBoost,
        non_linear_emul=non_linear_emul)
    # the accuracy rules shared by every wrapper: the overall boost
    # multiplies the cosmolike boost, and the integration accuracy and
    # the C_ell table length grow with it
    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(
        0, CLIntegrationAccuracy + abs(3*(CLAccuracyBoost - 1.0)))
    ci.init_ntable_lmax(int(_CONFIG["lmax"] + 20000*(CLAccuracyBoost - 1)))
    ci.init_photoz_conventions(
        int(_CONFIG["photoz_interpolation_type"]),
        int(_CONFIG["photoz_zmid_convention"]))
    # init_fpt_internal_boost comes first, as in the likelihood:
    # init_accuracy_boost sets the C-FAST-PT internal grid fraction
    # (internal_accuracyboost) to base x CLAccuracyBoost, where base is
    # the fraction it found at its first call in the process; a fraction
    # set after init_accuracy_boost would discard the boost
    ci.init_fpt_internal_boost(
        float(_CONFIG["internal_accuracyboost"]))
    ci.init_accuracy_boost(CLAccuracyBoost, int(CLIntegrationAccuracy))
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
                     chi=chi,
                     omegan2=omegan2)
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
    """Replaces None shear vectors with the fiducial ones.

    Arguments:
      M, shear_photoz_bias, A1, A2, BTA = the source-bin nuisance lists
          of a wrapper call (one entry per source bin), or None.

    Returns:
      the five lists in the same order, each None replaced by M_FID,
      SHEAR_PHOTOZ_FID, A1_FID, A2_FID or BTA_FID.
    """
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
    """Replaces None clustering vectors with the fiducial ones.

    Arguments:
      lens_photoz_bias, lens_photoz_stretch, B1, B2, B_MAG, B3nl, BK =
          the lens-bin nuisance lists of a wrapper call (one entry per
          lens bin), or None.

    Returns:
      the seven lists in the same order, each None replaced by its
      fiducial list (LENS_PHOTOZ_FID, LENS_STRETCH_FID, B1_FID, BMAG_FID,
      or ZEROS6 for B2, B3nl and BK).
    """
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
    ci.C_ss_tomo_limber, the spectra in the Limber approximation. The
    nuisance vectors default to the module fiducials when passed as
    None.

    Arguments:
      ell = 1D array of multipoles; the rest as in _set_state, with
      the shear group only (this wrapper never touches clustering).

    Returns:
      (EE, BB): two 3D arrays (n_ell, n_bin, n_bin), n_bin = source
      bins; only the pairs i <= j are filled, the other entries are 0.
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
    binning can change between calls without restarting the
    notebook's Python process (the Jupyter kernel); the binning
    arguments default to the configure()d values.

    Arguments:
      ntheta, theta_min_arcmin, theta_max_arcmin = the logarithmic
      angular binning (edges in arcmin), or None; the rest as in
      C_ss_tomo_limber.

    Returns:
      (theta, xi_plus, xi_minus): theta = the area-weighted bin centers
      in arcmin, xi 3D arrays (n_theta, n_bin, n_bin) with both orders
      of a source pair filled.
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
    call; blocks outside the selected probes never read theirs (a
    "xi" run ignores the clustering state).

    Arguments:
      galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
      galaxy_bias_b3nl, galaxy_bias_bk = the lens-bin lists B1, B2,
          B_MAG, B3nl and BK of _set_state, or None for the fiducials;
      PM = the point masses, or None for PM_FID; the rest as in
          _set_state (kmax defaults to 7.5 here, 10 elsewhere).

    Returns:
      float chi2 = (d - t)^T C^-1 (d - t) over the entries the mask
      keeps (d the data, t the theory vector).
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
# In the Limber approximation a multipole l reads P(k) at k = (l + 1/2)/chi
# along the line of sight, so a spectrum is an integral over ln k. The
# response d ln C/d ln k is the fraction of C(l) contributed per unit
# ln k, and the cumulative response R(k_max) integrates its absolute
# value over ln k up to ln k_max: they show which wavenumbers (h/Mpc) a
# multipole or an angle probes.
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

    Arguments:
      k = 1D array of wavenumbers in h/Mpc; ell = 1D array of
      multipoles; the rest as in C_ss_tomo_limber.

    Returns:
      (EE, BB) as ci.dlnC_ss_dlnk_tomo_limber returns them: two arrays
      (n_k, n_ell, n_bin, n_bin).
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

    Arguments:
      k = 1D array of wavenumbers in h/Mpc; the rest as in xi.

    Returns:
      (theta, dlnxip_dlnk, dlnxim_dlnk): theta in arcmin, the two
      responses arrays (n_k, n_theta, n_bin, n_bin).
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

    Arguments:
      k = 1D array of k_max values in h/Mpc; ell = 1D array of
      multipoles; the rest as in C_ss_tomo_limber.

    Returns:
      (EE, BB) as ci.rf_C_ss_tomo_limber returns them: two arrays
      (n_k, n_ell, n_bin, n_bin).
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

    Arguments:
      k = 1D array of k_max values in h/Mpc; the rest as in xi.

    Returns:
      (theta, rf_xip, rf_xim): theta in arcmin, the two responses
      arrays (n_k, n_theta, n_bin, n_bin).
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

    Shear state as in C_ss_tomo_limber (the CMB is a single source
    plane, so only the shear nuisances enter), then
    ci.C_ks_tomo_limber (Limber approximation).

    Arguments:
      ell = 1D array of multipoles; the rest as in C_ss_tomo_limber.

    Returns:
      2D array (n_ell, n_bin), one column per source bin.
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

    Arguments:
      ntheta, theta_min_arcmin, theta_max_arcmin = the angular binning,
      or None for the configure()d values; the rest as in
      C_ks_tomo_limber.

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

    Arguments:
      k = 1D array of wavenumbers in h/Mpc; ell = 1D array of
      multipoles; the rest as in C_ks_tomo_limber.

    Returns:
      array as ci.dlnC_ks_dlnk_tomo_limber returns it,
      (n_k, n_ell, n_bin).
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

    Arguments:
      k = 1D array of k_max values in h/Mpc; ell = 1D array of
      multipoles; the rest as in C_ks_tomo_limber.

    Returns:
      array as ci.rf_C_ks_tomo_limber returns it, (n_k, n_ell, n_bin).
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

    Arguments:
      k = 1D array of wavenumbers in h/Mpc; the rest as in w_ks.

    Returns:
      (theta, dlnwks_dlnk): theta in arcmin, the response array
      (n_k, n_theta, n_bin).
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

    Arguments:
      k = 1D array of k_max values in h/Mpc; the rest as in w_ks.

    Returns:
      (theta, rf_wks): theta in arcmin, the response array
      (n_k, n_theta, n_bin).
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


# ----------------------------------------------------------------------
# Baryonic feedback via the bfmt theory block
# ----------------------------------------------------------------------
def get_baryon_suppression(theory_options, point, z_grid, log10k_grid):
    """Return the suppression S(k, z) of the bfmt theory block on a grid.

    S = P(k) with baryonic feedback / P(k) without it. Builds a minimal
    Cobaya model (CAMB + bfmt + the likelihood "one", which returns
    ln L = 0 and only makes the model complete), requests the
    baryon_suppression product at the given grid (k in 1/Mpc, as the
    Cosmolike likelihoods send it; the block converts to h/Mpc
    internally), evaluates it at this module's fiducial cosmology, and
    returns {z: S array over k}.

    Arguments:
      theory_options = bfmt options dict, e.g. {"baryon_model": 2}.
      point   = {parameter name: value} for the method's feedback
                parameters, fixed in the model.
      z_grid  = redshifts of the evaluation grid.
      log10k_grid = log10 of the wavenumbers, read as k in 1/Mpc.

    Returns:
      {z: 1D S array over k}, one entry per z_grid value.
    """
    from cobaya.model import get_model
    # Cobaya model description: CAMB at this module's fiducial
    # cosmology (tau = 0.0543, a Planck 2018 value, only completes
    # CAMB's input; omch2 subtracts the massive-neutrino density),
    # bfmt with the caller's options and feedback parameters, and
    # debug = 50 (logging.CRITICAL: only critical messages print)
    info = {
        "likelihood": {"one": None},
        "theory": {
            # no "path" for camb: the session already imported it,
            # and cobaya accepts the loaded module as is
            "camb": {"extra_args": {"halofit_version": "takahashi",
                                    "dark_energy_model": "ppf"}},
            "bfmt": dict({"python_path": os.environ["ROOTDIR"]
                          + "/external_modules/code/baryon_suppression"},
                         **theory_options),
        },
        "params": dict({
            "As": {"value": As_1e9*1e-9},
            "ns": ns, "H0": H0, "mnu": mnu, "tau": 0.0543,
            "w": w,
            "ombh2": omegab*(H0/100)**2,
            "omch2": (omegam-omegab)*(H0/100)**2
                     - (mnu*(3.046/3)**0.75)/94.0708,
            "omegam": {"derived": True, "latex": r"\Omega_m"},
        }, **point),
        "debug": 50,
    }
    model = get_model(info)
    model.add_requirements({"baryon_suppression": {
        "z": z_grid, "k": np.power(10.0, log10k_grid)}})
    # every parameter is fixed, so the point to evaluate is the empty
    # dictionary; the call runs CAMB and bfmt once
    model.logposterior({})
    return model.provider.get_baryon_suppression()


def compute_probes(sup=None, ell=None):
    """Compute the fiducial cosmic-shear statistics, data vector and chi2.

    Returns the shear spectra C_ss, the correlation functions xi_+-,
    the CMB lensing x shear spectra C_ks and correlation w_ks, the
    masked data vector and its chi2 at this module's fiducial point,
    with optional baryonic suppression folded into the nonlinear
    power. Requires init_cosmolike(CLprobe=..., with_data=True): the
    probe selection decides which blocks of the data vector enter dv
    and chi2 (EXAMPLE_EVALUATE1 selects "xi", the cosmic-shear
    entries). sup = None computes the dark-matter-only prediction;
    otherwise sup is the {z: S array} dictionary from
    get_baryon_suppression, applied the way the Cosmolike likelihoods
    apply it: lnPNL[i :: len(z_grid)] += ln S(z_i).

    Every call rebuilds the interface state with the settings of
    get_chi2: CAMB with kmax = 7.5 (in 1/Mpc, CAMB's unit) and
    k_per_logint = 10, the C_ell tables at the configured lmax,
    accuracy boost 1 and integration level 0, every nuisance group at
    its fiducial value, and no tabulated hydro-simulation ratio. The
    call also re-runs init_binning with the configured angular binning
    (by default the dataset's 30 bins from 0.25 to 250 arcmin), because
    the real-space outputs and the data vector read the binning and a
    real-space wrapper called with another ntheta leaves its own
    binning behind. get_chi2 runs on the binning in effect, so right
    after init_cosmolike the no-feedback chi2 here equals the fiducial
    get_chi2() value.

    Arguments:
      sup = suppression dictionary on the CAMB interpolation grid
            (one entry per z of z_grid, each an array over the k of
            log10k_grid), or None.
      ell = multipoles of the returned harmonic spectra, or None
            for np.arange(25, 3000, 15).

    Returns:
      dict with
        ell         = the multipoles of C_ss and C_ks, [n_ell];
        C_ss        = the EE spectra, [n_ell, n_bin, n_bin], pairs
                      i <= j filled, the other entries 0;
        C_ks        = CMB lensing x shear spectra in the Limber
                      approximation, without the lensing-map filter,
                      [n_ell, n_bin];
        theta       = angular bin centers in arcmin, [n_theta];
        xip, xim    = xi_plus and xi_minus, [n_theta, n_bin, n_bin],
                      both orders of a pair filled;
        w_ks        = CMB lensing x shear correlation, with the
                      lensing-map filter, [n_theta, n_bin];
        dv          = the masked data vector, [n_data], zero where the
                      mask removes an entry;
        chi2        = (d - dv)^T C^-1 (d - dv) over the entries the
                      mask keeps, d the measured data vector and C its
                      covariance;
        ndata       = number of entries the mask keeps, the data
                      points chi2 runs over;
        z_grid      = z nodes of the CAMB power-spectrum tables;
        log10k_grid = log10 of their k nodes, with k in 1/Mpc, the
                      unit get_baryon_suppression takes (the notebooks
                      feed both grids to that function).

      legend: n_ell = len(ell), n_bin = 4 source bins, n_theta = the
      configured angular bins (30), n_data = 1,809 entries of the
      6x2pt layout.
    """
    if ell is None:
        ell = np.arange(25., 3000., 15.)
    # kmax = 7.5 (1/Mpc) and k_per_logint = 10: the CAMB settings of
    # get_chi2
    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_growth, z_interp_1D, chi,
     omegan2, lnPL_cb) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, kmax=7.5, k_per_logint=10,
        CAMBAccuracyBoost=1.0,
        non_linear_emul=_CONFIG["non_linear_emul"])
    # a private copy of ln P_nonlinear, modified in place below
    lnPNL = np.array(lnPNL, copy=True)
    if sup is not None:
        for i, z_val in enumerate(z_interp_2D):
            # every k row of redshift z_i sits at stride len(z) in
            # the flattened table, the layout set_cosmology expects;
            # sup[z_val] looks the redshift up by exact float equality,
            # which holds when sup was computed on this same z_grid
            lnPNL[i :: len(z_interp_2D)] += np.log(sup[z_val])
    # the C_ell tables follow the law of _set_state,
    # int(lmax + 20000 (CLAccuracyBoost - 1)), at CLAccuracyBoost = 1
    ci.init_ntable_lmax(int(_CONFIG["lmax"]))
    ci.init_photoz_conventions(
        int(_CONFIG["photoz_interpolation_type"]),
        int(_CONFIG["photoz_zmid_convention"]))
    # init_fpt_internal_boost comes first, as in the likelihood and in
    # _set_state, so the C-FAST-PT internal grid fraction is the
    # configured one even when this is the first init_accuracy_boost of
    # the process
    ci.init_fpt_internal_boost(
        float(_CONFIG["internal_accuracyboost"]))
    # accuracy boost 1 and integration level 0, the defaults of get_chi2
    ci.init_accuracy_boost(1.0, 0)
    # xi_pm_tomo, w_ks_tomo and the data vector read the angular binning;
    # the data vector and its mask are laid out on the dataset's binning,
    # which the configured one mirrors
    ci.init_binning(ntheta_bins=int(_CONFIG["ntheta"]),
                    theta_min_arcmin=float(_CONFIG["theta_min_arcmin"]),
                    theta_max_arcmin=float(_CONFIG["theta_max_arcmin"]))
    # the galaxy-bias model, selected before the bias amplitudes are set,
    # as _set_state does (init_cosmolike skips it for the shear-only
    # probe sets)
    ci.init_bias(bias_model=_CONFIG["bias_model"])
    ci.set_cosmology(omegam=omegam, H0=H0,
                     log10k_2D=log10k_interp_2D, z_2D=z_interp_2D,
                     lnP_linear=lnPL, lnP_nonlinear=lnPNL,
                     G=G_growth, z_G=z_growth,
                     z_1D=z_interp_1D, chi=chi,
                     omegan2=omegan2)
    ci.set_nuisance_shear_calib(M=M_FID)
    ci.set_nuisance_shear_photoz(bias=SHEAR_PHOTOZ_FID)
    ci.set_nuisance_clustering_photoz(bias=LENS_PHOTOZ_FID,
                                      stretch=LENS_STRETCH_FID)
    ci.set_nuisance_bias(B1=B1_FID, B2=ZEROS6, B_MAG=BMAG_FID,
                         B3nl=ZEROS6, BK=ZEROS6)
    ci.set_nuisance_ia(A1=A1_FID, A2=A2_FID, B_TA=BTA_FID)
    ci.set_point_mass(PMV=PM_FID)
    # no tabulated hydro-simulation ratio: sup is the only feedback
    ci.reset_bary_struct()
    # tmp = the BB spectra, not returned
    (C_ss, tmp) = ci.C_ss_tomo_limber(l=ell)
    C_ks = ci.C_ks_tomo_limber(l=ell)
    (xip, xim) = ci.xi_pm_tomo()
    wks = ci.w_ks_tomo()
    theta = np.array(ci.get_binning_real_space())
    dv = np.array(ci.compute_data_vector_masked())
    chi2 = ci.compute_chi2(dv)
    # the mask holds one 0/1 flag per entry, and the blocks of the
    # probes init_probes left out are already 0 in it: its sum is the
    # number of data points chi2 runs over
    ndata = int(np.sum(ci.get_mask()))
    return {"ell": ell, "C_ss": C_ss, "C_ks": C_ks,
            "theta": theta, "xip": xip, "xim": xim, "w_ks": wks,
            "dv": dv, "chi2": chi2, "ndata": ndata,
            "z_grid": z_interp_2D,
            # get_baryon_suppression takes k in 1/Mpc, but the CAMB helper
            # returns this grid in h/Mpc: convert here, at the one place
            # that links the two, so S(k) is evaluated at the physical k.
            "log10k_grid": log10k_interp_2D + np.log10(H0/100.0)}
