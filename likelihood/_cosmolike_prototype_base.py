"""Base class of every desy1xplanck likelihood: from cobaya to cosmolike.

The project combines DES galaxy surveys with Planck CMB lensing. Its
default data set, Y3xPlanckPR4.dataset, holds the DES Y3 x Planck PR4
6x2pt measurement: one data vector of 1809 numbers in six blocks,

    ss  cosmic shear xi_+ and xi_-          entries    0 -  599
    gs  galaxy-galaxy lensing gamma_t       entries  600 - 1319
    gg  galaxy clustering w(theta)          entries 1320 - 1499
    gk  lens galaxies x CMB lensing         entries 1500 - 1679
    ks  source shear x CMB lensing          entries 1680 - 1799
    kk  CMB lensing auto-spectrum bands     entries 1800 - 1808

built from 4 source (shear) redshift bins and 6 lens bins of MagLim, the
magnitude-limited DES lens galaxy sample. The bins are defined with
photometric redshifts (photo-z), so each bin carries nuisance parameters
for photo-z errors; each lens bin has a shift and a stretch (DES_DZ_L<i>,
DES_DZ2_L<i>; see set_lens_related). A mask file marks which entries
enter the chi2 (the scale cuts); the theory vector holds zeros at the
entries the mask removes.

cobaya is the sampler framework that runs the analysis: it reads the
yaml file, builds the theory codes (cobaya calls them theory blocks:
CAMB, or the emulators of the EXAMPLE_EMUL2 examples) and one object per
likelihood block. Each module of this folder (cosmic_shear.py,
combo_6x2pt.py, ...) defines a class that only names its probes;
everything else is inherited from the class below, whose methods cobaya
calls in this order:

  initialize          once: reads the .dataset file, builds the
                      redshift and wavenumber grids, configures the
                      compiled cosmolike library (n(z), mask, covariance,
                      CMB-lensing filters, accuracy settings).
  get_requirements    once: tells cobaya what the theory codes compute.
  logp                at every parameter point: hands the cosmology and
                      the nuisance parameters to cosmolike, computes the
                      theory data vector and returns -chi2/2.

The compiled library is imported as ci (cosmolike_desy1xplanck_interface,
built from interface/interface.cpp). It keeps its state in C global
variables: every ci.init_* and ci.set_* call replaces part of that state,
and the ci.compute_* calls read whatever was set last.
"""

# In Python 3 the three features named below are already the default, so
# this import changes nothing. Python accepts a from __future__ import only
# at the top of a module, after the docstring and comments.
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools
from collections.abc import Mapping

# cobaya: the base class of likelihoods that read a .dataset file, and the
# error type cobaya reports cleanly. getdist's IniFile reads the
# `key = value` lines of a .dataset file.
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

# EuclidEmulator2: emulated nonlinear boost P_nl/P_lin (non_linear_emul = 1)
import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  """Print the wall-clock time that a `with timer(label):` block takes.

  A context manager is the object a with statement uses: the code before
  yield runs when the block starts and the code after yield runs when the
  block ends. The contextmanager decorator builds one from this generator
  function. timer is a debugging aid: wrap a call in
  `with timer("set_cosmology"):` to print its duration.

  Arguments:
    label = text printed in front of the elapsed time.

  Returns:
    a context manager; at the end of the block it prints
    "<label>: <seconds>s" with four decimals.
  """
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

# The compiled cosmolike library of this project (interface/ holds the .so
# that scripts/compile_desy1xplanck.sh builds; scripts/start_desy1xplanck.sh
# puts interface/ on PYTHONPATH).
import cosmolike_desy1xplanck_interface as ci

# OpenMP threads for cosmolike's parallel loops, read once when this module
# is imported: the OMP_NUM_THREADS environment variable, 1 when it is unset.
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """Wrap fn so that cosmolike gets its OpenMP thread count back first.

    A decorator is a function that receives a function and returns a
    replacement: writing @with_omp_threads above a method definition
    stores the returned wrapper under the method's name. Each call of the
    wrapper runs ci.set_omp_threads(COSMOLIKE_OMP_THREADS) and then the
    original method with the same arguments (*args and **kwargs collect
    the positional and the keyword arguments and pass them on unchanged).

    The reason: cosmolike's loops run in parallel with OpenMP and take the
    number of threads from omp_get_max_threads(), which starts at
    OMP_NUM_THREADS. Some Python libraries call omp_set_num_threads(1)
    without saying so; that call changes the setting for the whole
    process, so every later cosmolike loop would run on one core.
    ci.set_omp_threads also keeps the BLAS calls inside cosmolike on one
    thread. functools.wraps copies the name and the docstring of fn onto
    the wrapper, so error messages and help() still show the original.

    Arguments:
      fn = the method to wrap (set_cosmo_related, set_source_related,
           set_lens_related and get_datavector below).

    Returns:
      the wrapper function: same arguments and return value as fn.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

# Prefix of every nuisance-parameter name of this project (DES_M1,
# DES_DZ_L1, DES_B1_1, ...); cosmic_shear.py and combo_2x2pt.py import it.
survey = "DES"

class _cosmolike_prototype_base(DataSetLikelihood):
  """Shared machinery of every desy1xplanck likelihood class.

  cobaya's DataSetLikelihood stores each option of the likelihood's yaml
  block as an attribute of the object: the default value from the yaml
  file next to the subclass (combo_6x2pt.yaml, ...), replaced by the
  user's value when the user's yaml sets one (self.accuracyboost,
  self.IA_model, ...). The module docstring lists the methods cobaya
  calls and when. The switches that change the path through this class:

    use_emulator     0 = CAMB computes the power spectra and distances;
                     2 = trained emulators replace CAMB for them (the
                     EXAMPLE_EMUL2 examples), cosmolike computes the rest;
                     1 = emulators of whole data-vector blocks (cosmic
                     shear, ggl, wtheta): initialize and get_requirements
                     carry a branch for it, but no method of this class
                     reads those emulator results.
    non_linear_emul  1 = EuclidEmulator2 boost below z = 10, the theory
                     code's nonlinear P(k) above; 2 = the theory code's
                     nonlinear P(k) everywhere.
    IA_model         the model of intrinsic alignments (IA: galaxy shapes
                     aligned by the local tidal field, which mimic
                     lensing): 0 = NLA (nonlinear alignment, linear in
                     the tidal field), 1 = TATT (tidal alignment and
                     tidal torquing, adds second-order terms).
    IA_code          who computes the one-loop (next-to-leading order in
                     perturbation theory) IA and galaxy-bias spectra that
                     TATT and the nonlinear bias need: 0 = FAST-PT built
                     into cosmolike (C), 1 = the Python FAST-PT theory
                     block. FAST-PT evaluates those integrals with FFTs.
    external_nz_modeling, external_baryon_suppression, add_baryons_on_dv,
    create_baryon_pca, use_baryon_pca: see initialize and the methods that
                     read them.

  The comments of the yaml files describe the accuracy settings.
  """

  @classmethod
  def get_modified_defaults(cls, defaults, input_options={}):
    """Apply the yaml option `fixed_params` to the default parameters.

    A class method (the @classmethod decorator) receives the class itself
    as cls instead of an object, so cobaya can call it before any
    likelihood object exists. cobaya calls it when it reads the defaults of a
    combination (its yaml file, e.g. combo_3x2pt_ks_gk_kk.yaml), before it
    merges them with the user's yaml. The parameters of a combination come
    from `params: !defaults [params_source, params_lens_maglim]`, and the
    `!defaults` tag builds the whole `params` mapping from those files, so
    the same yaml cannot change one entry of it. A combination that fixes
    some of these parameters lists them under `fixed_params` instead
    (combo_3x2pt_ks_gk_kk fixes the point masses, which act only on
    galaxy-galaxy lensing). Each entry replaces the parameter's default
    info with the cobaya merge rule: a value drops prior, ref and proposal
    and keeps the other keys (the latex label). A user yaml can override
    `fixed_params` like any other option of the likelihood. Combinations
    without `fixed_params` keep their defaults unchanged.

    Arguments:
      defaults = the combination's default options (dict, `params`
                 included), changed in place
      input_options = the user's options for this likelihood (dict)

    Returns:
      defaults, with each parameter of `fixed_params` replaced.
    """
    fixed = input_options.get("fixed_params", defaults.get("fixed_params"))
    params = defaults.get("params") or {}
    for p, info in (fixed or {}).items():
      old = params.get(p)
      new = {}
      if isinstance(old, Mapping):
        # keep every key of the default info except the sampling ones
        for key, value in old.items():
          if key not in ("prior", "ref", "proposal"):
            new[key] = value
      if isinstance(info, Mapping):
        new.update(info)
      else:
        new["value"] = info
      params[p] = new
    if params:
      defaults["params"] = params
    return defaults

  def initialize(self, probe):
    """Read the .dataset file and configure cosmolike for one probe set.

    cobaya calls initialize() once, when it builds the likelihood; each
    subclass calls this method with its probe name (cosmic_shear.py passes
    "xi", combo_6x2pt.py passes "6x2pt"). The .dataset file (data_file,
    inside the folder path) is a small text file of `key = value` lines
    that name the data-vector, covariance, mask and n(z) files and give
    the bin counts, the angular binning and the CMB-lensing settings.

    The order of the ci calls matters: ci.initial_setup resets every
    global variable of cosmolike to its default, and the calls after it
    fill in this likelihood's choices.

    Arguments:
      probe = the probe combination, a key of the table in cosmolike's
              init_probes (cosmolike/generic_interface.cpp): here "xi",
              "2x2pt", "2x2pt_ss_sg", "2x2pt_ss_sk", "3x2pt",
              "3x2pt_ks_gk_kk", "3x2pt_ss_sk_sk" or "6x2pt".

    Returns:
      nothing; the grids and file names become attributes of self, and
      the global state of the compiled library is replaced.

    Raises:
      LoggedError when pk_z_refinement is not a positive integer.
    """
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    self.lens_ntomo = ini.int("lens_ntomo")
    self.source_ntomo = ini.int("source_ntomo")
    self.ntheta = ini.int("n_theta")
    self.theta_min_arcmin = ini.float("theta_min_arcmin")
    self.theta_max_arcmin = ini.float("theta_max_arcmin")

    # ------------------------------------------------------------------------
    # z_interp_1D: the redshift nodes of the comoving distance chi(z) and of
    # the growth table handed to cosmolike (set_cosmo_related). Three
    # uniform blocks: [0, 3) holds the galaxy n(z) (the source n(z) file
    # ends at z = 2.99); [3, 50.1) covers the CMB-lensing line-of-sight
    # integrals, which cosmolike starts at z = 40 (limits.a_min = 1/41 in
    # cosmolike/structs.c); [1070, 1100] brackets z = 1090, where cosmolike
    # places the CMB source plane (chi(a = 1/1091) in g_cmb,
    # cosmolike/redshift_spline.c). At accuracyboost 1, tmp = 1250 and the
    # blocks hold 1000, 500 and 125 nodes.
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # z_interp_2D: the z nodes of the P(k, z) tables handed to cosmolike.
    # cosmolike interpolates linearly in z between exactly these nodes (it
    # computes the node index directly on each uniform block, without a
    # search and without regridding). Linear interpolation leaves a
    # sawtooth-shaped error of order dz^2 that vanishes at the nodes, so
    # two grids that do not share nodes differ by the full amplitude of
    # that error. A node count that grows smoothly with the boost (for
    # example min(120 + 20*boost, 250)) moves every node when the boost
    # changes: the sawtooth shifts instead of shrinking, which gives
    # order-unity chi2 jitter in roman_kl's clustering vector and smaller
    # shifts in this project that do not converge as the boost grows.
    #
    # The factor m = 2^ceil(log2(boost)) refines each uniform block by an
    # integer factor and keeps its endpoints, so (a) every block stays
    # uniform and cosmolike keeps its two-block direct indexing, (b) the
    # nodes of a coarser grid are a subset of the nodes of every finer
    # grid, so a higher boost is a true refinement (the error falls like
    # 1/m^2), and (c) m = 1 gives the 140 nodes that CAMB itself is asked
    # for (z_interp_2D_camb below). The low block multiplies its node
    # count (endpoint=False, spacing 3/(105 m)); the high block multiplies
    # its number of intervals (endpoint=True: 35 nodes are 34 intervals,
    # so the block holds 34 m + 1 nodes).
    #
    # The tables end at z = 49.99: the CMB-lensing integrals need P(k, z)
    # up to z = 40 (see z_interp_1D above), and the hybrid power-spectrum
    # emulator of use_emulator = 2 predicts P(k, z) only up to z = 50. The
    # galaxy-only probes would need much less.
    #
    # pk_z_refinement multiplies m on top of the accuracy boost. A
    # Fourier-space data vector reads P(k, z) at fixed multipoles, where
    # the linear-interpolation error does not average out as it does in
    # real space: the roman_fourier 3x2pt chi2 moves by Delta chi2 = 0.25,
    # 0.030 and 0.002 from m = 1 to m = 2, 4 and 8, the roman_real and
    # lsst_y1 ones by Delta chi2 <= 0.004 from m = 1 to 2.
    #
    # getattr(self, name, default) returns the yaml option, or default when
    # the likelihood's yaml file does not declare the option at all.
    zref = getattr(self, "pk_z_refinement", 1)
    if not (float(zref) == int(zref) and int(zref) >= 1):
      raise LoggedError(self.log, "pk_z_refinement = %s: must be a positive "
                        "integer", zref)
    # m = the smallest power of two >= accuracyboost (1 for a boost of at
    # most 1), capped at 16, then multiplied by pk_z_refinement
    m = int(min(2**np.ceil(np.log2(max(1.0, self.accuracyboost))), 16))
    m = m*int(zref)
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False), 
                                       np.linspace(3.0,49.99,34*m + 1)),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    # CAMB's transfer module caps the number of requested redshifts at
    # 256, so the list handed to CAMB through the Pk_interpolator
    # requirement stays at this boost-independent 140-node grid (the
    # m = 1 grid above). The denser nested nodes only re-evaluate the
    # smooth spline in z that cobaya builds through these redshifts when
    # the cosmolike tables are filled, so raising the boost refines
    # exactly the table resampling that produced the jitter, and the
    # CAMB side never exceeds its cap.
    self.z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False), 
                                            np.linspace(3.0,49.99,35)),axis=0)

    # log10 of the wavenumbers of the P(k, z) tables, k in 1/Mpc (the unit
    # of CAMB and cobaya): 1250 + 250*accuracyboost nodes (1500 at boost 1)
    # from k = 1.0e-5 to 100/Mpc. set_cosmo_related converts k to h/Mpc,
    # cosmolike's unit.
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    # initial_setup resets cosmolike's global state; init_probes switches on
    # the blocks of this probe combination (the mask entries of the other
    # blocks become zero); init_binning sets the n_theta angular bins
    # between theta_min_arcmin and theta_max_arcmin (arcmin).
    ci.initial_setup()
    ci.init_probes(possible_probes=self.probe)
    ci.init_binning(int(self.ntheta), self.theta_min_arcmin, self.theta_max_arcmin)

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    # How the n(z) table files are read: interpolation 0 = cubic spline,
    # 1 = linear, 2+ = Steffen (monotone, cannot dip below zero); the z
    # column holds 0 = left bin edges (Z_LOW), 1 = sample points (Z_MID).
    ci.init_photoz_conventions(
        interpolation_type=int(getattr(self, "photoz_interpolation_type", 0)),
        zmid_convention=int(getattr(self, "photoz_zmid_convention", 0)))

    # The C FAST-PT convolution grid as a fraction of its output table (1 =
    # the two grids are equal). It must be set before init_accuracy_boost:
    # the first init_accuracy_boost call of the process stores the fraction
    # it finds as the base that every later call multiplies by the boost.
    ci.init_fpt_internal_boost(
        internal_boost=float(getattr(self, "internal_accuracyboost", 1.0)))

    # The exact (non-Limber) projection evaluates its line-of-sight
    # integrals with FFTLog, a fast Fourier transform on a grid uniform in
    # ln(chi); nonlimber_accuracyboost refines that chi grid on top of the
    # accuracy boost (narrow lens bins need it: see
    # init_nonlimber_accuracy_boost in cosmolike).
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(getattr(self, "nonlimber_accuracyboost", 1.0)))

    # The Limber approximation evaluates P(k, z) only at k = (l + 1/2)/chi,
    # which fails at low multipoles for narrow redshift kernels. For
    # galaxy-galaxy lensing, 1 = Limber at every multipole (default) and
    # 0 = the exact projection below l = 150. Galaxy clustering has the
    # same choice with default 0: the narrow lens n(z) break the
    # approximation there.
    ci.init_adopt_limber_gs(
        adopt_limber_gs=int(getattr(self, "adopt_limber_gs", 1)))

    ci.init_adopt_limber_gg(
        adopt_limber_gg=int(getattr(self, "adopt_limber_gg", 0)))
    # 0 = perturbative galaxy bias, 1 = halo-model galaxy power from a halo
    # occupation distribution (HOD, the mean number of galaxies per halo
    # of mass M); always set, so a model never inherits the previous
    # model's value
    ci.init_include_HOD_GX(
        include_HOD_GX=int(getattr(self, "include_HOD_GX", 0)))
    # 0 = the init_IA model, 1 = halo-model IA (Fortuna et al. 2021)
    ci.init_include_halo_IA(
        include_halo_IA=int(getattr(self, "include_halo_IA", 0)))
    # Halo statistics (sigma(M), the mass function, the halo bias) use the
    # cold dark matter + baryon spectrum P_cb. The emulators of
    # use_emulator = 2 provide no P_cb, so get_neutrino_inputs derives it
    # from P_lin with the small-scale ratio; the log says so once.
    if self.use_emulator == 2:
      self.log.info("Halo P_cb uses P_lin/(1 - f_nu)^2 because the "
                    "emulators have no cb spectrum (an approximation; "
                    "see get_neutrino_inputs)")

    # CMB lensing x galaxy cross-correlations (gk, ks): the multipole range
    # lmin_kx..lmax_kx of the sum over l that turns C_l into w(theta), the
    # beam of the CMB lensing map (fwhm_kx, full width at half maximum in
    # arcmin) and the HEALPix pixel window (healpix_win_func_kx_file): the
    # filters of the CMB lensing map, applied to the cross spectra.
    ci.init_cmb_cross_correlation(
        lmin = ini.int("lmin_kx"),
        lmax = ini.int("lmax_kx"), 
        fwhm = ini.float("fwhm_kx"), 
        healpixwin_filename = ini.relativeFileName('healpix_win_func_kx_file')
      )
    # CMB lensing auto-spectrum (kk): nbp_kk bandpowers, each a weighted sum
    # of C_l^kk over l = lminbp_kk..lmaxbp_kk (one row of the binning-matrix
    # file per band) plus the per-band offset of offset_kk_file. alpha is
    # the Hartlap factor (N_sim - N_band - 2)/(N_sim - 1), with N_sim =
    # hartlap_nvar_kk simulations behind the kk covariance and N_band =
    # nbp_kk: the inverse of a covariance estimated from N_sim simulations
    # is biased high by 1/alpha (Hartlap et al. 2007), so cosmolike scales
    # the kk block of the inverse covariance by alpha.
    nbins = ini.int("nbp_kk")
    nvar = ini.float("hartlap_nvar_kk")
    ci.init_cmb_auto_bandpower(
        nbins  = nbins,
        lmin = ini.int("lminbp_kk"),
        lmax = ini.int("lmaxbp_kk"),
        binning_matrix = ini.relativeFileName('binmat_kk_file'),
        theory_offset = ini.relativeFileName('offset_kk_file'),
        alpha = (nvar - nbins - 2.0)/(nvar - 1.0))

    # use_emulator = 1: the data-vector blocks come from emulators, so
    # cosmolike needs only the n(z), the data files and coarse tables (the
    # point-mass term of gamma_t, PM, needs the distances alone).
    if self.use_emulator == 1:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)  
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      # CAMB (use_emulator = 0) or the power-spectrum emulators (2):
      # cosmolike computes every block. lmax is the highest multipole of
      # the C_l tables summed into the real-space correlation functions
      # (arcminute angles need l of order 10^4 to 10^5); is_linear=False
      # makes cosmolike use the nonlinear P(k) of set_cosmology.
      ci.init_ntable_lmax(lmax=int(self.lmax))
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      ci.init_cosmo_runmode(is_linear=False)

      # external_nz_modeling = 1: read the n(z) tables into numpy arrays
      # (self.lens_nz, self.source_nz) that the likelihood sends again at
      # every point (set_lens_related, set_source_related), so a user
      # function can modify them; 0: cosmolike reads the n(z) files once.
      if self.external_nz_modeling: 
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      
      # covariance, mask (1 keeps an entry, 0 removes it) and data vector
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

      if (int(self.IA_model) == 0) and (int(self.IA_code) == 1):
        # NLA needs no one-loop IA tables, so IA_code = 1 (the Python
        # FAST-PT theory block) falls back to cosmolike's C FAST-PT, and
        # get_requirements, which runs after initialize, does not ask
        # cobaya for the IA_PS and bias_PS results either.
        self.IA_code = 0
      # IA_redshift_evolution sets how the IA amplitudes depend on redshift;
      # 3 (the yaml default) is the power law A ((1 + z)/(1 + z_0))^eta
      # with a fixed pivot redshift z_0, A = DES_A1_1 and eta = DES_A1_2
      # (and the pair DES_A2_1, DES_A2_2 for TATT's tidal-torquing term).
      ci.init_IA(ia_model = int(self.IA_model), 
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      # bias_model: one code per galaxy-bias term, in the order (b1, b2,
      # bs2, b3, bmag); 0 = one free amplitude per lens bin (DES_B1_<i>,
      # ...), 1 for b2, bs2 or b3 = derived from b1 (cosmolike/bias.c: b2
      # from the Lazeyras et al. 2016 fit, bs2 = -4/7 (b1 - 1), b3 =
      # b1 - 1). The yaml default [0, 0, 0, 1, 0] derives b3 from b1. The
      # probes without lens galaxies (xi, 3x2pt_ss_sk_sk, 2x2pt_ss_sk) need
      # no galaxy bias.
      if self.probe not in ("xi", "3x2pt_ss_sk_sk", "2x2pt_ss_sk"):
        ci.init_bias(bias_model=self.bias_model)

      # non_linear_emul = 1: build EuclidEmulator2 once; set_cosmo_related
      # asks it for the boost B(k, z) = P_nl/P_lin at every point
      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()

      # Baryonic-feedback options:
      #   external_baryon_suppression  the bfmt theory block multiplies the
      #       nonlinear P(k, z) by its suppression factor S(k, z)
      #       (set_cosmo_related);
      #   create_baryon_pca  this run computes the baryon principal
      #       components (PCs) from the hydrodynamical simulations listed in
      #       baryon_pca_select_sims and writes them to filename_baryon_pca
      #       (internal_get_datavector);
      #   add_baryons_on_dv  cosmolike multiplies the matter power spectrum
      #       by the ratio measured in the simulation which_bsims_add_on_dv
      #       (stored in all_sims_hdf5_file), to make contaminated data;
      #   use_baryon_pca  see below.
      # The lines below resolve conflicts without a message:
      # external_baryon_suppression turns off use_baryon_pca and
      # add_baryons_on_dv; create_baryon_pca turns off
      # external_baryon_suppression and use_baryon_pca and skips
      # add_baryons_on_dv.
      if self.external_baryon_suppression:
          self.use_baryon_pca = False
          self.add_baryons_on_dv = False

      if self.create_baryon_pca:
        self.external_baryon_suppression = False
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          self.external_baryon_suppression = False
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=self.allsims)

    # use_baryon_pca: marginalize over baryonic feedback with npcs principal
    # components of the data vector (columns of baryon_pca_file): the theory
    # vector gains sum_i Q_i PC_i at the kept entries, with the amplitudes
    # Q_i = DES_BARYON_Q1..Q4 sampled (internal_get_datavector).
    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    """Tell cobaya which quantities the theory codes must compute.

    cobaya calls this once, after initialize. The returned dictionary
    maps a quantity name to its options (None = no options); cobaya
    passes the requests to the theory blocks of the yaml (CAMB, or the
    emulators of the EXAMPLE_EMUL2 examples) and, at every evaluation,
    serves the results through self.provider.

    Requested for use_emulator = 0 (CAMB) and 2 (emulators):
      As, H0, omegam, omegab    parameters read in set_cosmo_related.
      Pk_interpolator           P(k, z) at the z_interp_2D_camb redshifts
                                up to k_max = kmax_boltzmann*accuracyboost
                                (in 1/Mpc), linear and nonlinear, of total
                                matter (delta_tot); with CAMB also of cold
                                dark matter + baryons (delta_nonu).
      comoving_radial_distance  chi(z) at the z_interp_1D nodes, in Mpc.
      mnu (emulators) or omnuh2 (CAMB)
                                the massive-neutrino density
                                (get_neutrino_inputs).
      mnu, w, wa                also for EuclidEmulator2
                                (non_linear_emul = 1).
      IA_PS, bias_PS            one-loop IA and galaxy-bias spectra of the
                                Python FAST-PT block (IA_code = 1).
      baryon_suppression        S(k, z) of the bfmt block, CAMB path with
                                external_baryon_suppression only.
      Cl                        the CAMB path also asks for the TT
                                spectrum up to l = 0 (see the note on
                                that line).
    use_emulator = 1 requests the data-vector emulator results
    (cosmic_shear, ggl, wtheta) of the probe, plus H0 and chi(z) when the
    probe has galaxy-galaxy lensing (the point-mass term).

    Returns:
      dict {quantity name: options}. With use_emulator = 1 only the
      probes "xi", "3x2pt", "xi_gg", "xi_ggl" and "2x2pt" have an entry;
      for the other probes the method returns None.
    """
    if self.use_emulator == 1:
      if self.probe == "xi":
        return {
          'cosmic_shear': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'cosmic_shear': None,
          'wtheta': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.use_emulator == 2:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
      }
      # IA_code = 1: the Python FAST-PT theory block computes the one-loop
      # IA (IA_PS) and galaxy-bias (bias_PS) power spectra
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # EuclidEmulator2 takes these parameters as inputs
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # mnu gives Omega_nu h^2 on this path (get_neutrino_inputs)
      _requirements_["mnu"] = None
      return _requirements_
    else:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
      # external_baryon_suppression: the bfmt theory block computes the
      # suppression S(k, z) (matter power with baryonic feedback divided by
      # the dark-matter-only one) exactly at the nodes of the cosmolike
      # tables: z_interp_2D and the k of log10k_interp_2D, in 1/Mpc. The
      # block converts k to h/Mpc where its models need that unit.
      if self.external_baryon_suppression:
          _requirements_["baryon_suppression"] = {
              "z": self.z_interp_2D,
              "k": np.power(
                  10.0, self.log10k_interp_2D
              ),
          }
      # IA_code = 1: the Python FAST-PT theory block computes the one-loop
      # IA (IA_PS) and galaxy-bias (bias_PS) power spectra
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      # EuclidEmulator2 takes these parameters as inputs
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2) and, for
      # the cold dark matter + baryon halo field, the linear P_cb
      # (get_neutrino_inputs)
      _requirements_["omnuh2"] = None
      # Ask for total matter (delta_tot) and for cold dark matter + baryons
      # (delta_nonu, CAMB's name for "no neutrinos"): the likelihood and the
      # halo-model functions of the interface that a notebook can call
      # directly (ci.sigma2, ci.hb1nu, ...) find P_cb even when the data
      # vector does not use it. CAMB computes both from one run of its
      # transfer functions.
      _requirements_["Pk_interpolator"]["vars_pairs"] = [
        ("delta_tot", "delta_tot"),
        ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    """Hand this point's power spectra, growth and distances to cosmolike.

    Runs once per likelihood evaluation (internal_get_datavector). It
    reads the theory results through cobaya's provider (self.provider),
    converts them to cosmolike's units and grids, and calls
    ci.set_cosmology.

    Units: the theory codes work in 1/Mpc and Mpc^3, cosmolike in h/Mpc
    and (Mpc/h)^3, so log10 k shifts by -log10(h), ln P by +ln(h^3), and
    chi(z) is multiplied by h (Mpc -> Mpc/h).

    Shape flow of each power spectrum:

      PKL.logP(z_interp_2D, k)        [n_z, n_k]   ln P, Mpc^3
        -> flatten(order='F')         [n_z*n_k]    entry iz + n_z*ik
        -> + ln(h^3)                  ln P in (Mpc/h)^3

      legend: n_z = len(z_interp_2D), n_k = len(log10k_interp_2D);
              order='F' (Fortran order) runs the z index fastest.

    The nonlinear spectrum is the theory code's (non_linear_emul = 2) or,
    below z = 10, P_lin times the EuclidEmulator2 boost (non_linear_emul =
    1). With use_emulator = 1 only the distances are set (the point-mass
    term of gamma_t needs them).

    Returns:
      nothing; the cosmology state of the compiled library is replaced.

    Raises:
      an error when non_linear_emul is neither 1 nor 2.
    """
    h = self.provider.get_param("H0")/100.0
    if not (self.use_emulator == 1):
      # PKL: cobaya's interpolator of the linear total-matter P(k, z), a
      # bicubic spline in z and ln k through the theory code's output,
      # extended as a straight line in ln P vs ln k down to k = 1e-6/Mpc
      # and up to 250*accuracyboost/Mpc
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmin=1e-6,
                                               extrap_kmax=2.5e2*self.accuracyboost)
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # EuclidEmulator2 covers z <= 10 and 8.73e-3 <= k <= 9.4 h/Mpc, so
        # the boost B = P_nl/P_lin is requested at the table redshifts below
        # 10 on that k range (10^-2.0589 to 10^0.973 h/Mpc)
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        # ln B at the table's k nodes, converted to h/Mpc: linear in log10 k,
        # extended as a straight line beyond 9.4 h/Mpc; below 8.73e-3 h/Mpc
        # ln B = 0 (P_nl = P_lin on those large scales). tmp has one row per
        # table redshift below 10 and one column per k node.
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        # lnbt [n_z, n_k]: ln B on the rows z < 10, zero on the others
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # The theory code's nonlinear P(k, z) (with CAMB, the halofit_version
        # of the yaml) covers every redshift ...
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # ... and the rows z < 10 take ln P_lin + ln B instead. reshape(...,
        # order='F') undoes the Fortran-order flattening to [n_z, n_k],
        # np.where picks each row from the first array where the condition
        # holds and from the second elsewhere, and ravel(order='F')
        # flattens the result back.
        lnPNL = np.where((self.z_interp_2D<10)[:,None], 
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')+lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", non_linear_emul)

      # G(z) = D(z)(1 + z), with D the linear growth factor from
      # D(z)/D(0) = sqrt(P_lin(k, z)/P_lin(k, 0)) at k = growth_k; G tends
      # to a constant in matter domination, where D grows like a = 1/(1+z).
      # G is tabulated on the dense 1D grid, cut at the last P(k) node:
      # cosmolike reads G linearly in z, and on the coarse 2D grid
      # (dz ~ 0.03) the linear read misses D by up to 9e-5 and the growth
      # rate f = 1 - (1+z) dlnG/dz (the slope of the table) by 1%; on the
      # 1D grid (dz = 0.003) by 1e-6 and 0.2%. PKL is a spline in z through
      # the theory code's output redshifts, so the dense grid asks the
      # theory code for no extra redshifts (about 0.1 ms per evaluation).
      # The table is divided by G at the last z_2D node, z_norm = 49.99
      # (z_growth ends below it); cosmolike's growfac divides by G(0), so
      # D(z=0) = 1.
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      # growth_k (default 0.05/Mpc) is a sub-horizon scale. At k = 5e-4/Mpc
      # (about 2 H0/c) CAMB's dark-energy perturbations change the growth
      # by 0.5-0.9% at w != -1 (z = 0.5 to 2), while every use of G in
      # cosmolike (IA amplitudes, the D^4 of the one-loop terms,
      # sigma(M, z), the growth rate f) concerns sub-horizon modes; with
      # 0.06 eV neutrinos the growth varies by 0.03% above 0.05/Mpc (the
      # measurements are in external_modules/code/cosmolike_core/.claude/
      # skills/cosmolike-dev/references/growth_factor_measurements.md).
      growth_k = float(getattr(self, "growth_k", 0.05))
      G_growth = np.sqrt(PKL.P(z_growth,growth_k)/PKL.P(0,growth_k))*(1+z_growth)
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,growth_k)/PKL.P(0,growth_k))*(1+z_norm)
      # external_baryon_suppression: multiply P_nl by the bfmt block's
      # S(k, z). The block returns {z: S at the k nodes} and sets S = 1
      # outside the calibrated (k, z) range of its model. With the
      # Fortran-order flattening, lnPNL[i::n_z] (every n_z-th entry from
      # entry i) is the k row of z node i, so ln S is added row by row.
      # A redshift missing from the result is skipped with a warning, and
      # an error while reading the result is logged and the suppression
      # skipped: the evaluation then continues without feedback.
      if self.external_baryon_suppression:
        try:
          supp_dict = self.provider.get_result("baryon_suppression")
          self.log.info(
            "Applying baryon suppression: %d redshifts from theory block",
            len(supp_dict),
          )

          for i, z_val in enumerate(self.z_interp_2D):
            if z_val in supp_dict:
              sup_array = supp_dict[z_val]
              lnbt_baryon = np.log(sup_array)
              lnPNL[i :: self.len_z_interp_2D] += lnbt_baryon
              self.log.debug(
                  "Applied baryon suppression at z=%.3f: "
                  "min_sup=%.6f, max_sup=%.6f",
                  z_val,
                  sup_array.min(),
                  sup_array.max(),
              )
            else:
              self.log.warning(
                  "baryon_suppression dict does not contain z=%.3f; skipping",
                  z_val,
              )
        except Exception as e:
            self.log.error(
                "Failed to retrieve baryon suppression from theory block: %s; "
                "skipping baryon suppression",
                str(e),
            )

      # the massive neutrinos: Omega_nu h^2 and, for the cold dark matter
      # + baryon halo field, the linear P_cb (get_neutrino_inputs)
      (omegan2, lnPL_cb) = self.get_neutrino_inputs(lnPL=lnPL, h=h)

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        omegab=self.provider.get_param("omegab"),
        omegan2=omegan2,
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_linear_cb=lnPL_cb,
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_G=z_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
      
      # IA_code = 1: hand cosmolike the one-loop IA and galaxy-bias tables
      # of the Python FAST-PT block. FPTIA has 12 rows of N entries: row 10
      # (FPTIA[-2]) is the k grid in h/Mpc, the other rows are spectra in
      # (Mpc/h)^3; flatten(order='C') writes it row after row, as
      # set_IA_PS expects. This must come after ci.set_cosmology, which
      # renews cosmology.random, the key cosmolike's caches use to detect a
      # new cosmology.
      if int(self.IA_code) == 1:
        FPTIA, FPTIA_kcut  = self.provider.get_IA_PS()
        FPTbias, sigma4    = self.provider.get_bias_PS()
        FPT_kmin, FPT_kmax = FPTIA[-2,0], FPTIA[-2,-1]
        
        ci.set_IA_PS(PS=FPTIA.flatten(order='C'), 
                     kmin=FPT_kmin, 
                     kmax=FPT_kmax, 
                     cutoff=FPTIA_kcut, 
                     N=len(FPTIA[0]))
        
        ci.set_bias_PS(PS=FPTbias.flatten(order='C'), 
                       kmin=FPT_kmin, 
                       kmax=FPT_kmax, 
                       cutoff=FPTIA_kcut, 
                       sigma4=sigma4, 
                       N=len(FPTIA[0]))
    else:
      # use_emulator = 1: only chi(z), which the point-mass term of gamma_t
      # needs (in Mpc/h, like set_cosmology's chi)
      ci.set_distances(
        z=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def get_neutrino_inputs(self, lnPL, h):
    """Return the massive-neutrino inputs of ci.set_cosmology.

    omegan2 is Omega_nu h^2 of the massive neutrinos today, part of
    omegam. Massive neutrinos stream freely out of halos, so cosmolike's
    halo statistics use the cold dark matter + baryon spectrum P_cb: the
    halo variance sigma^2(M, z) (the variance of the density field
    smoothed over the radius that encloses mass M) is computed from P_cb,
    and the mass-radius relation and the mass-function density use
    rho_crit (Omega_m - Omega_nu). Total matter stays the input of
    lensing and of the separate total-matter variance.

    lnPL_cb is ln P_cb on the same (k, z) grid and in the same units as
    lnPL. Both spectra are always provided, so the halo-model functions
    that a notebook calls directly on the interface (ci.sigma2,
    ci.hb1nu, ...) work even after an evaluation whose data vector
    needed no halo quantities.

    The two theory paths:
      CAMB (use_emulator = 0): omegan2 is CAMB's omnuh2 and P_cb its
        ("delta_nonu", "delta_nonu") linear spectrum, read like P_lin
        (get_requirements asks for both).
      emulators (use_emulator = 2): the emulators take no neutrino
        parameter (they were trained at mnu = 0.06 eV) and have no cb
        spectrum. omegan2 = mnu (3.046/3)^0.75/94.0708, the neutrino
        density the yaml's omegach2 subtracts, and
        P_cb = P_lin/(1 - f_nu)^2 with f_nu = omegan2/(omegam h^2): the
        ratio of the two spectra at wavenumbers far above the neutrino
        free-streaming wavenumber, where the neutrinos do not cluster.
        On cluster scales this is an approximation; its measured size
        is in projects/des_cluster/README.md.

    Arguments:
      lnPL = ln P_lin [(Mpc/h)^3], flattened as set_cosmology's
             lnP_linear (Fortran order: k index slow, z index fast)
      h    = H0/100

    Returns:
      (omegan2, lnPL_cb): a float and a numpy array of lnPL's shape.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

    if self.use_emulator == 2:
      # P_cb/P_lin = 1/(1 - f_nu)^2 at wavenumbers where the neutrinos do
      # not cluster (there delta_m = (1 - f_nu) delta_cb)
      f_nu = omegan2/(self.provider.get_param("omegam")*h*h)
      lnPL_cb = lnPL - 2.0*np.log(1.0 - f_nu)
    else:
      # the same k extrapolation, (z, k) grid, flattening and units as
      # lnPL in set_cosmo_related
      PKL_cb = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                 nonlinear=False,
                                                 extrap_kmin=1e-6,
                                                 extrap_kmax=2.5e2*self.accuracyboost)
      k_grid = np.power(10.0, self.log10k_interp_2D)
      lnPL_cb = PKL_cb.logP(self.z_interp_2D, k_grid).flatten(order='F')
      lnPL_cb = lnPL_cb + np.log(h**3)
    return (omegan2, lnPL_cb)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_source_related(self, **params):
    """Hand this point's source-sample nuisance parameters to cosmolike.

    Each family is a list with one entry per source bin i = 1..ntomo,
    read from params by name:

      DES_M<i>       multiplicative shear calibration m_i: the shear of
                     bin i is multiplied by (1 + m_i)
      DES_DZ_S<i>    photo-z shift of source bin i (redshift units)
      DES_A1_<i>, DES_A2_<i>, DES_BTA_<i>
                     intrinsic-alignment parameters; with
                     IA_redshift_evolution = 3 cosmolike reads only the
                     first two entries of A1 and A2 (amplitude and
                     redshift power-law index) and the first of B_TA

    A name absent from params counts as 0, so a likelihood whose yaml
    does not declare a family sends zeros.

    Arguments:
      params = the parameters of this point, by name (the ** collects the
               keyword arguments into a dictionary).

    Returns:
      nothing; the source nuisance state of cosmolike is replaced.
    """
    ntomo = self.source_ntomo
    # The inner comprehension builds the names DES_M1, DES_M2, ... and the
    # outer one their values, params.get(name, 0) being 0 for a name that
    # params does not contain; the same pattern serves every list below.
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling: 
        # external_nz_modeling: the source n(z) read at initialization is
        # sent again at every point, so a user function of the nuisance
        # parameters can modify it first (for example to add outliers).
        # copy() returns a new array, so a modification never reaches
        # self.source_nz; the commented line below marks where such a
        # function goes, before ci.set_source_sample.
        source_nz_local = self.source_nz.copy()

        #source_nz_local = f(source_nz_local, nuisance parameters)

        ci.set_source_sample(source_nz_local)

        # the photo-z shifts DES_DZ_S<i> still apply on top of the n(z)
        # sent above; a user model that includes them can drop this call
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_lens_related(self, **params):
    """Hand this point's lens-sample nuisance parameters to cosmolike.

    Each family is a list with one entry per lens bin i = 1..ntomo, read
    from params by name (the list pattern of set_source_related):

      DES_PM<i>      point mass of lens bin i, in 10^13 M_sun/h: adds a
                     term proportional to 1/theta^2 to gamma_t, the shear
                     of mass inside the smallest scales that the model
                     does not describe
      DES_B1_<i>     linear galaxy bias b1
      DES_B2_<i>     quadratic bias b2 (one-loop terms)
      DES_BMAG_<i>   magnification-bias coefficient
      DES_B3NL_<i>, DES_BK_<i>
                     third-order and nonlocal bias amplitudes
      DES_DZ_L<i>    photo-z shift dz_i of lens bin i
      DES_DZ2_L<i>   photo-z stretch s_i of lens bin i

    The lens photo-z model shifts and stretches each n(z) about the mean
    redshift zbar_i of the tabulated bin (cosmolike/redshift_spline.c):

      n_i(z) -> n_i((z - dz_i - zbar_i)/s_i + zbar_i) / s_i

    so s_i = 1 leaves the width unchanged and the division keeps the
    integral of n_i equal to 1.

    A name absent from params counts as 0, except DES_B1_<i>, which counts
    as 1. A stretch of 0 is not "no stretch" (that is 1): a likelihood
    that computes lens observables must declare DES_DZ2_L<i>, as
    params_lens_maglim.yaml does. The probes without lens observables
    (2x2pt_ss_sk, 3x2pt_ss_sk_sk) send these zeros but never use them.

    Arguments:
      params = the parameters of this point, by name.

    Returns:
      nothing; the lens nuisance state of cosmolike is replaced.
    """
    ntomo = self.lens_ntomo
    ci.set_point_mass(
      PMV = [params.get(p, 0) for p in [survey+"_PM"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]],
        B3nl=[params.get(p,0) for p in [survey+"_B3NL_"+str(i+1) for i in range(ntomo)]],
        BK=[params.get(p,0) for p in [survey+"_BK_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # external_nz_modeling: the lens n(z) read at initialization is
        # sent again at every point, so a user function of the nuisance
        # parameters can modify it first (for example to add outliers).
        # copy() returns a new array, so a modification never reaches
        # self.lens_nz; the commented line below marks where such a
        # function goes, before ci.set_lens_sample.
        lens_nz_local = self.lens_nz.copy()

        #lens_nz_local = f(lens_nz_local, nuisance parameters)

        ci.set_lens_sample(lens_nz_local)

        # the photo-z shifts and stretches still apply on top of the n(z)
        # sent above; a user model that includes them can drop this call
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]],
          stretch=[params.get(p,0) for p in [survey+"_DZ2_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]],
          stretch=[params.get(p,0) for p in [survey+"_DZ2_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    """Return the log-likelihood -chi2/2 of one theory data vector.

    cosmolike computes chi2 = (d - t)^T C^-1 (d - t) over the entries the
    mask keeps, with d the data vector of the .dataset file, t the theory
    vector and C^-1 the masked inverse covariance (with the Hartlap
    factor on its kk block, see initialize).

    Arguments:
      datavector = the theory data vector, full length (1809 entries for
                   Y3xPlanckPR4.dataset), zero at the removed entries.

    Returns:
      float, -chi2/2.
    """
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params):
    """Return the log-likelihood of one parameter point.

    cobaya calls logp at every evaluation of the model.

    Arguments:
      params = every input parameter of this likelihood, by name.

    Returns:
      float, -chi2/2 (compute_logp).
    """
    datavector = self.internal_get_datavector(**params)
    return self.compute_logp(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):        
    """Return the theory data vector of one point as a numpy array.

    For notebooks and scripts; the sampler goes through logp. With
    use_emulator = 1 the emulator call is commented out below and the
    method returns the array form of the number 0.0, not a data vector.

    Arguments:
      params = the parameters of this point, by name.

    Returns:
      numpy float64 array, full data-vector length (internal_get_datavector).
    """
    if self.use_emulator == 1:
      #dv = self.internal_get_datavector_emulator(**params)
      dv = 0.0
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    """Compute the masked theory data vector of one parameter point.

    The cosmology goes to cosmolike first, then the lens nuisance
    parameters (every probe except cosmic shear, "xi") and the source
    ones; cosmolike then evaluates the blocks the probe selection keeps.
    The baryon options change the last step:

      create_baryon_pca  compute the baryon PCs from the simulations of
                         baryon_pca_select_sims, write them to
                         filename_baryon_pca, return the plain vector;
      use_baryon_pca     add sum_i Q_i PC_i, Q_i = DES_BARYON_Q<i>;
      otherwise          the plain vector.

    Arguments:
      params = the parameters of this point, by name.

    Returns:
      list of floats, full data-vector length, zero at the entries the
      mask removes.

    Side effects:
      with create_baryon_pca, writes filename_baryon_pca; with
      print_datavector, writes print_datavector_file, two columns: the
      entry index and the value (the format pair '%d', '%1.8e').
    """
    self.set_cosmo_related()
    if self.probe != "xi":
        self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else:  
      datavector = ci.compute_data_vector_masked()

    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      # the comma makes fmt the tuple ('%d', '%1.8e'): one format per
      # column, the index as an integer, the value with 8 decimals
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
