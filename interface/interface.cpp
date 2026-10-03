#ifdef _OPENMP
#include <omp.h>
#endif
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>
#include <map>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

#include <boost/algorithm/string.hpp>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/baryons.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

#include <carma.h>
#include <armadillo>
#include "cosmolike/generic_interface.hpp"
#include "cosmolike/cosmo2D_wrapper.hpp"
#include "cosmolike/cosmo2D_scuts_wrapper.hpp"
#include "cosmolike/halo_wrapper.hpp"

PYBIND11_MODULE(cosmolike_desy1xplanck_interface, m)
{
  m.doc() = "CosmoLike Interface for DESY3 x Planck 6x2pt Module";

  // --------------------------------------------------------------------
  // INIT FUNCTIONS
  // --------------------------------------------------------------------
  m.def("init_ntable_lmax",
      &cosmolike_interface::init_ntable_lmax,
      "Init accuracy and sampling Boost (may slow down Cosmolike a lot)",
      (py::arg("lmax") = 75000).none(false)
    );

  m.def("init_ntable_ell_internal",
      &cosmolike_interface::init_ntable_ell_internal,
      "Coarse exact-quadrature ell nodes of the C_ss/C_gs/C_gk/C_ks "
      "tables and of the scale-cut tables' ell axis, cubic-spline "
      "upsampled to N_ell; 0 = exact per-node quadrature",
      (py::arg("nell_internal") = 192).none(false)
    );

  m.def("init_ntable_dcx_dlnk_nlnk_internal",
      &cosmolike_interface::init_ntable_dcx_dlnk_nlnk_internal,
      "Coarse exact ln k nodes of the scale-cut machinery (the dC "
      "tables and the dlnxi/dlnw caches), cubic upsampled; the ell "
      "axis follows N_ell_internal; default 128 of the 256 grid, "
      "0 = exact",
      (py::arg("nlnk_internal") = 128).none(false)
    );

  m.def("init_ntable_nm_internal",
      &cosmolike_interface::init_ntable_nm_internal,
      "Coarse exact nodes of the sigma^2(M) halo-model table, "
      "cubic-spline upsampled to N_M in ln sigma^2; 0 = exact",
      (py::arg("nm_internal") = 192).none(false)
    );

  m.def("init_ntable_halo_ia_lmax",
      &cosmolike_interface::init_ntable_halo_ia_lmax,
      "Highest multipole of the halo-model IA satellite profile: 2, 4 "
      "or 6 (Fortuna et al. 2021: 6)",
      (py::arg("halo_ia_lmax") = 6).none(false)
    );

  m.def("sigma2",
      &cosmolike_interface::compute_sigma2,
      "Halo-model mass variance sigma^2(M) at a = 1 from the cached "
      "lobe-summed table; M in M_sun/h (diagnostic)",
      (py::arg("M")).none(false)
    );

  m.def("init_accuracy_boost",
      &cosmolike_interface::init_accuracy_boost,
      "Init accuracy and sampling Boost (may slow down Cosmolike a lot)",
      (py::arg("accuracy_boost") = 1.0).none(false),
      (py::arg("integration_accuracy") = 0).none(false)
    );

  m.def("init_photoz_conventions",
      &cosmolike_interface::init_photoz_conventions,
      "Set the n(z) interpolation type (0: cspline, 1: linear, 2+: steffen) "
      "and the n(z) file z-column convention (0: Z_LOW, 1: Z_MID)",
      (py::arg("interpolation_type") = 0).none(false),
      (py::arg("zmid_convention") = 0).none(false)
    );

  m.def("init_fpt_internal_boost",
      &cosmolike_interface::init_fpt_internal_boost,
      "Set the C-FAST-PT internal (convolution) grid as a fraction of the "
      "output table (1.0 = grids equal, the exact legacy path)",
      (py::arg("internal_boost") = 1.0).none(false)
    );

  m.def("init_nonlimber_accuracy_boost",
      &cosmolike_interface::init_nonlimber_accuracy_boost,
      "Refine the non-Limber FFTLog chi grid (Ntable.NL_Nchi) on top of the "
      "accuracy boost (1.0 = the accuracy boost alone)",
      (py::arg("nonlimber_boost") = 1.0).none(false)
    );

  m.def("init_adopt_limber_gs",
      &cosmolike_interface::init_adopt_limber_gs,
      "Galaxy-galaxy lensing: 1 = Limber at every multipole (default), "
      "0 = non-Limber below limits.LMAX_NOLIMBER",
      (py::arg("adopt_limber_gs") = 1).none(false)
    );

  m.def("init_adopt_limber_gg",
      &cosmolike_interface::init_adopt_limber_gg,
      "Galaxy clustering: 0 = non-Limber below limits.LMAX_NOLIMBER, "
      "1 = Limber at every multipole",
      (py::arg("adopt_limber_gg") = 0).none(false)
    );

  m.def("init_include_HOD_GX",
      &cosmolike_interface::init_include_HOD_GX,
      "Galaxy probes: 0 = perturbative galaxy bias (default), 1 = "
      "halo-model (HOD) galaxy power from halo.c (needs adopt_limber_gg = 1)",
      (py::arg("include_HOD_GX") = 0).none(false)
    );

  m.def("init_include_halo_IA",
      &cosmolike_interface::init_include_halo_IA,
      "Cosmic shear and ggl: 0 = the init_IA model (default), 1 = "
      "halo-model IA (Fortuna et al. 2021; NLA, Limber gs, no HOD)",
      (py::arg("include_halo_IA") = 0).none(false)
    );

  m.def("init_halo_matter_field",
      &cosmolike_interface::init_halo_matter_field,
      "Halo-model density field of sigma(M) and dn/dM: 0 = total matter "
      "(default), 1 = cold dark matter + baryons (needs set_cosmology's "
      "omegan2 and lnP_linear_cb)",
      (py::arg("halo_matter_field") = 0).none(false)
    );

  m.def("init_baryons_contamination",
      py::overload_cast<std::string, std::string>(
         &cosmolike_interface::init_baryons_contamination),
      "Init data vector contamination (on the matter power spectrum) with baryons",
      py::arg("sim").none(false),
      py::arg("allsims").none(false)
    );

  m.def("init_bias", 
      &cosmolike_interface::init_bias, 
      "Set the bias modeling choices",
      py::arg("bias_model").none(false),
      py::return_value_policy::move
    );

  m.def("init_binning",
      &cosmolike_interface::init_binning_real_space,
      "Init Bining related variables",
      py::arg("ntheta_bins").none(false).noconvert(),
      py::arg("theta_min_arcmin").none(false),
      py::arg("theta_max_arcmin").none(false)
    );

  m.def("init_binning_fourier",
      [](int nells, int lmin, int lmax) {
        cosmolike_interface::init_binning_fourier(nells, lmin, lmax, -1);
      },
      "Init Bining related variables",
      py::arg("nells").none(false).noconvert(),
      py::arg("lmin").none(false),
      py::arg("lmax_shear").none(false)
    );

  m.def("init_cosmo_runmode",
      &cosmolike_interface::init_cosmo_runmode,
      "Init Run Mode (should we force the matter power spectrum to be linear)",
      py::arg("is_linear").none(false)
    );

  m.def("init_cmb_cross_correlation",
      &cosmolike_interface::init_cmb_cross_correlation,
      "Init scale cuts (fourier) and experiment binning for the CMB cross correlation",
      py::arg("lmin").none(false).noconvert(),
      py::arg("lmax").none(false).noconvert(),
      py::arg("fwhm").none(false),
      py::arg("healpixwin_filename").none(false)
    );

  m.def("init_cmb_auto_bandpower",
      &cosmolike_interface::init_cmb_auto_bandpower,
      "Init scale cuts (fourier & bandpower) and cov correctionfor the CMB kk",
      py::arg("nbins").none(false).noconvert(),
      py::arg("lmin").none(false).noconvert(),
      py::arg("lmax").none(false).noconvert(),
      py::arg("binning_matrix").none(false),
      py::arg("theory_offset").none(false),
      py::arg("alpha").none(false)
    );

  m.def("init_data_real",
      [](std::string cov, std::string mask, std::string data) {
        using namespace cosmolike_interface;
        init_data_Mx2pt_N<0,6>(cov, mask, data, {0, 1, 2, 3, 4, 5});
      },
      "Load covariance matrix, mask (vec of 0/1s) and data vector",
      py::arg("COV").none(false),
      py::arg("MASK").none(false),
      py::arg("DATA").none(false)
    );

  m.def("init_IA",
      &cosmolike_interface::init_IA_fastpt,
      "Init IA related options",
      py::arg("ia_model").none(false).noconvert(),
      py::arg("ia_redshift_evolution").none(false).noconvert(),
      py::arg("ia_code").none(false).noconvert()
    );

  m.def("init_probes",
      &cosmolike_interface::init_probes,
      "Init Probes (cosmic shear or 2x2pt or 3x2pt...)",
      py::arg("possible_probes").none(false)
    );

  m.def("initial_setup",
      &cosmolike_interface::initial_setup,
      "Initialize Cosmolike Variables to their Default Values"
    );

  m.def("init_redshift_distributions_from_files",
      &cosmolike_interface::init_redshift_distributions_from_files,
      "Init lens and source n(z) from files",
      py::arg("lens_multihisto_file").none(false),
      py::arg("lens_ntomo").none(false).noconvert(),
      py::arg("source_multihisto_file").none(false),
      py::arg("source_ntomo").none(false).noconvert()
    );

  m.def("init_survey_parameters",
      &cosmolike_interface::init_survey,
      "Init Survey Parameters",
      py::arg("surveyname").none(false),
      py::arg("area").none(false),
      py::arg("sigma_e").none(false)
    );

  m.def("read_redshift_distributions",
    &cosmolike_interface::read_redshift_distributions_from_files,
    "Read n(z) lens and source from files (same way as old cosmolike)",
    py::arg("lens_multihisto_file").none(false),
    py::arg("lens_ntomo").none(false).noconvert(),
    py::arg("source_multihisto_file").none(false),
    py::arg("source_ntomo").none(false).noconvert(),
    py::return_value_policy::move
  );
  
  m.def("set_IA_PS",
      &cosmolike_interface::set_IA_PS,
      "Set FPTIA if FASTPT is called",
      py::arg("PS").none(false),
      py::arg("kmin").none(false),
      py::arg("kmax").none(false),
      py::arg("cutoff").none(false),
      py::arg("N").none(false)
    );

  m.def("set_bias_PS",
      &cosmolike_interface::set_bias_PS,
      "Set FPTbias if FASTPT is called",
      py::arg("PS").none(false),
      py::arg("kmin").none(false),
      py::arg("kmax").none(false),
      py::arg("cutoff").none(false),
      py::arg("sigma4").none(false),
      py::arg("N").none(false)
    );

  m.def("init_lens_sample_size",
      &cosmolike_interface::set_lens_sample_size,
      "Set the lens number of tomo bins",
      py::arg("Ntomo").none(false).noconvert()
    );

  m.def("init_source_sample_size",
      &cosmolike_interface::set_source_sample_size,
      "Set the source number of tomo bins",
      py::arg("Ntomo").none(false).noconvert()
    );

  m.def("init_ntomo_powerspectra",
    &cosmolike_interface::init_ntomo_powerspectra,
    "Set the number of power spectra"
  );

  // --------------------------------------------------------------------
  // SET FUNCTIONS
  // --------------------------------------------------------------------
  m.def("set_omp_threads",
    [](int n) {
#ifdef _OPENMP
      if (n > 0) { omp_set_num_threads(n); }
#else
      (void) n;
#endif
    },
    pybind11::arg("n"),
    "Set the OpenMP thread count for cosmolike's internal parallel regions. "
    "Must be called before any compute_* function if you've set because some "
    "Python libraries silently call omp_set_num_threads(1)");

  m.def("set_distances",
      [](arma::Col<double> z, 
         arma::Col<double> chi)
      {
        spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_distances");
        using namespace cosmolike_interface;
        set_distances(z, chi);
        spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_distances");
      },
      "Set Distance (Cosmology)",
       py::arg("z").none(false),
       py::arg("chi").none(false),
       py::return_value_policy::move
    );

  m.def("set_cosmology",
      [](const double omega_matter,
         const double hubble,
         arma::Col<double> io_log10k_2D,
         arma::Col<double> io_z_2D, 
         arma::Col<double> io_lnP_linear,
         arma::Col<double> io_lnP_nonlinear,
         arma::Col<double> io_G,
         arma::Col<double> io_z_1D,
         arma::Col<double> io_chi,
         const double omega_baryon,
         std::vector<double> io_z_G,
         const double omega_nu_h2,
         std::vector<double> io_lnP_linear_cb)
      {
        spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_cosmology");
        using namespace cosmolike_interface;
        set_cosmological_parameters(omega_matter, omega_baryon, hubble,
                                    omega_nu_h2);
        set_linear_power_spectrum(io_log10k_2D,io_z_2D,io_lnP_linear);
        // the linear P_cb (cold dark matter + baryons) on the grid of
        // lnP_linear, after it: sigma^2(M) reads it under
        // init_halo_matter_field(1). An empty list removes the table of
        // the previous call, so a stale P_cb is never read.
        if (io_lnP_linear_cb.empty()) {
          clear_linear_power_spectrum_cb();
        }
        else {
          set_linear_power_spectrum_cb(io_log10k_2D, io_z_2D,
                                       arma::Col<double>(io_lnP_linear_cb));
        }
        set_non_linear_power_spectrum(io_log10k_2D,io_z_2D,io_lnP_nonlinear);
        // growfac reads G linearly in z: the likelihood samples G on its
        // dense 1D grid (z_G) instead of the coarse z_2D grid of the power
        // spectra (whose size CAMB's transfer redshifts cap), because the
        // halo model, the IA amplitudes and f_growth inherit the error of
        // that linear read. Without z_G, G is sampled on z_2D.
        if (io_z_G.empty()) {
          set_growth(io_z_2D, io_G);
        }
        else {
          set_growth(arma::Col<double>(io_z_G), io_G);
        }
        set_distances(io_z_1D,io_chi);
        spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_cosmology");
      },
      "Set Cosmological Parameters, Distance, Matter Power Spectrum, Growth "
      "Factor, and the massive-neutrino density and linear P_cb of the halo "
      "model",
       py::arg("omegam").none(false),
       py::arg("H0").none(false),
       py::arg("log10k_2D").none(false),
       py::arg("z_2D").none(false),
       py::arg("lnP_linear").none(false),
       py::arg("lnP_nonlinear").none(false),
       py::arg("G").none(false),
       py::arg("z_1D").none(false),
       py::arg("chi").none(false),
       py::arg("omegab") = 0.0,
       py::arg("z_G") = std::vector<double>(),
       // Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2)
       py::arg("omegan2") = 0.0,
       // ln P_cb [(Mpc/h)^3], flattened as lnP_linear; empty = none
       py::arg("lnP_linear_cb") = std::vector<double>(),
       py::return_value_policy::move
    );

  m.def("set_baryon_pcs",
      [](arma::Mat<double> eigenvectors) {
        spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_baryon_pcs");
        cosmolike_interface::BaryonScenario::get_instance().set_pcs(eigenvectors);
        spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_baryon_pcs");
      },
      "Load baryonic principal components from numpy array",
       py::arg("eigenvectors").none(false)
    );

  m.def("set_nuisance_ia",
      &cosmolike_interface::set_nuisance_IA,
      "Set nuisance Intrinsic Aligment (IA) amplitudes",
      py::arg("A1").none(false),
      py::arg("A2").none(false),
      py::arg("B_TA").none(false),
      py::return_value_policy::move
    );

  m.def("set_nuisance_bias",
      &cosmolike_interface::set_nuisance_bias_fastpt,
      "Set nuisance Bias Parameters",
      py::arg("B1").none(false),
      py::arg("B2").none(false),
      py::arg("B_MAG").none(false),
      py::arg("B3nl").none(false),
      py::arg("BK").none(false),
      py::return_value_policy::move
    );

  m.def("set_nuisance_shear_calib",
      &cosmolike_interface::set_nuisance_shear_calib,
      "Set nuisance shear calibration amplitudes",
      py::arg("M").none(false),
      py::return_value_policy::move
    );

  m.def("set_nuisance_shear_photoz",
      &cosmolike_interface::set_nuisance_shear_photoz,
      "Set nuisance shear photo-z bias amplitudes",
      py::arg("bias").none(false),
      py::return_value_policy::move
    );

  m.def("set_nuisance_clustering_photoz",
      [](arma::Col<double> CP, arma::Col<double> CPS) {
        using namespace cosmolike_interface;
        set_nuisance_clustering_photoz(CP);
        set_nuisance_clustering_photoz_stretch(CPS);
      },
      "Set nuisance clustering shear photo-z bias & stretch amplitudes",
      py::arg("bias"),
      py::arg("stretch"),
      py::return_value_policy::move
    );

  m.def("set_point_mass",
      [](arma::Col<double> PM) {
        cosmolike_interface::PointMass::get_instance().set_pm_vector(PM);
      },
      "Set the point mass amplitudes",
      py::arg("PMV").none(false)
    );

  m.def("set_log_level_debug", 
      []() {
        spdlog::set_level(spdlog::level::debug);
      },
      "Set the SPDLOG level to debug"
    );

  m.def("set_log_level_info", 
      []() {
        spdlog::set_level(spdlog::level::info);
      },
      "Set the SPDLOG level to info"
    );

  m.def("set_lens_sample",
      &cosmolike_interface::set_lens_sample,
      "Set the lens n(z) from a numpy n(z) histogram",
      py::arg("nofz").none(false)
    );

  m.def("set_source_sample",
      &cosmolike_interface::set_source_sample,
      "Set the source n(z) from a numpy n(z) histogram",
      py::arg("nofz").none(false)
    );

  // --------------------------------------------------------------------
  // reset FUNCTIONS
  // --------------------------------------------------------------------
  
  m.def("reset_bary_struct",
      &reset_bary_struct,
       "Set the Baryon Functions to not contaminate the MPS w/ Baryon effects"
    );

  // --------------------------------------------------------------------
  // COMPUTE FUNCTIONS (relevant for emulators)
  // --------------------------------------------------------------------
  m.def("compute_add_fpm_3x2pt_real_any_order",
      [](arma::Col<double> dv, 
         const int force_exclude_pm)->std::vector<double> 
      {
        using namespace cosmolike_interface;
        arma::Col<double> res;
        if (force_exclude_pm == 1) {
          res = compute_add_calib_and_set_mask_Mx2pt_N<0,6,0>(dv,{0, 1, 2, 3, 4, 5});
        } 
        else {
          res = compute_add_calib_and_set_mask_Mx2pt_N<0,6,1>(dv,{0, 1, 2, 3, 4, 5});
        }
        return arma::conv_to<std::vector<double>>::from(res);
      },
      "Add fast shear calibration parameters to the theoretical data vector.",
      py::arg("datavector").none(false),
      py::arg("force_exclude_pm").none(false),
      py::return_value_policy::move
    );

  m.def("compute_add_fpm_3x2pt_real_any_order_with_pcs",
      [](arma::Col<double> dv, 
         arma::Col<double> Q, 
         const int force_exclude_pm)->std::vector<double> 
      {
        using namespace cosmolike_interface;
        using stlvec = std::vector<double>;
        arma::Col<double> res;
        if (force_exclude_pm == 1) {
          res = compute_add_calib_and_set_mask_Mx2pt_N<0,6,0>(dv,{0,1,2,3,4,5});
        } 
        else {
          res = compute_add_calib_and_set_mask_Mx2pt_N<0,6,1>(dv,{0,1,2,3,4,5});
        }
        return arma::conv_to<stlvec>::from(compute_add_baryons_pcs(Q,res));
      },
      "Add fast shear calibration parameters to the theoretical data vector.",
      py::arg("datavector").none(false),
      py::arg("Q").none(false),
      py::arg("force_exclude_pm").none(false),
      py::return_value_policy::move
    );

  m.def("compute_data_vector_3x2pt_real_sizes",
      []()->std::vector<int> {
        using namespace cosmolike_interface;
        using namespace arma;
        using stlvec = std::vector<int>;
        return conv_to<stlvec>::from(compute_data_vector_Mx2pt_N_sizes<0,6>());
      },
      "Returns the data vector sizes of each 2pt correlation function",
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // COMPUTE FUNCTIONS
  // --------------------------------------------------------------------
  m.def("compute_data_vector_masked",
      []()->std::vector<double> {
        using namespace cosmolike_interface;
        using stlvec = std::vector<double>;
        return arma::conv_to<stlvec>::from(compute_Mx2pt_N_masked<0,6>({0,1,2,3,4,5}));
      },
      "Compute theoretical data vector. Masked dimensions are filled w/ zeros",
      py::return_value_policy::move
    );

  m.def("compute_data_vector_masked_with_baryon_pcs",
      [](std::vector<double> Q)->std::vector<double> {
        using namespace cosmolike_interface;
        using stlvec = std::vector<double>;
        arma::Col<double> res = compute_Mx2pt_N_masked<0,6>({0,1,2,3,4,5});
        return arma::conv_to<stlvec>::from(compute_add_baryons_pcs(Q,res));
      },
      "Compute theoretical data vector, including contributions from baryonic"
      " principal components. Masked dimensions are filled w/ zeros",
      py::arg("Q").none(false),
      py::return_value_policy::move
    );

  m.def("compute_chi2",
      [](arma::Col<double> datavector) {
        return cosmolike_interface::IP::get_instance().get_chi2(datavector);
      },
      "Compute $\\chi^2$ given a theory data vector input",
      py::arg("datavector").none(false),
      py::return_value_policy::move
    );

  m.def("compute_baryon_pcas",
      [](std::string scenarios, std::string allsims) {
        using namespace cosmolike_interface;
        BaryonScenario::get_instance().set_scenarios(allsims, scenarios);
        return compute_baryon_pcas_Mx2pt_N<0,6>({0,1,2,3,4,5});
      },
      "Compute baryonic principal components given a list of scenarios" 
      "that contaminate the matter power spectrum",
      py::arg("scenarios").none(false),
      py::arg("allsims").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // Theoretical Cosmolike Functions
  // --------------------------------------------------------------------
  m.def("get_binning_real_space",
      // Why return an STL vector?
      // The conversion between STL vector and python np array is cleaner
      // arma:Col is cast to 2D np array with 1 column (not as nice!)
      []()->std::vector<double> {
        arma::Col<double> res = cosmolike_interface::get_binning_real_space();
        return arma::conv_to<std::vector<double>>::from(res);
      },
      "Get real space binning (theta bins)"
    );

  m.def("get_gs_redshift_bins",
      &cosmolike_interface::gs_bins,
      "Get galaxy-galaxy lensing redshift binning"
    );

  m.def("xi_pm_tomo",
      &cosmolike_interface::xi_pm_tomo_cpp,
      "Compute cosmic shear (real space) data vector at all tomographic"
      " and theta bins"
    );

  m.def("w_gammat_tomo",
      &cosmolike_interface::w_gammat_tomo_cpp,
      "Compute galaxy-galaxy lensing (real space) data vector at all"
      " tomographic and theta bins",
      py::return_value_policy::move
    );

  m.def("w_gg_tomo",
      &cosmolike_interface::w_gg_tomo_cpp,
      "Compute galaxy-galaxy clustering (real space) data vector at all"
      " tomographic and theta bins",
      py::return_value_policy::move
    );

  m.def("w_ks_tomo",
      &cosmolike_interface::w_ks_tomo_cpp,
      "Compute CMB lensing-shear (real space) data vector at all"
      " tomographic and theta bins",
      py::return_value_policy::move
    );

  m.def("C_ss_tomo_limber",
      py::overload_cast<const double, const int, const int>(
        &cosmolike_interface::C_ss_tomo_limber_cpp
      ),
      "Compute shear-shear (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert()
    );

  m.def("C_ss_tomo_limber",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_ss_tomo_limber_cpp),
      "Compute shear-shear (fourier - limber) data vector at all tomographic"
      " bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

      m.def("C_gs_tomo_limber",
      py::overload_cast<const double, const int, const int>(
        &cosmolike_interface::C_gs_tomo_limber_cpp),
      "Compute shear-position (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("nl").none(false).noconvert(),
      py::arg("ns").none(false).noconvert()
    );

  m.def("C_gs_tomo_limber",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_gs_tomo_limber_cpp),
      "Compute shear-position (fourier - limber) data vector at all tomographic"
      " bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

      m.def("C_gg_tomo_limber",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_gg_tomo_limber_cpp),
      "Compute position-position (fourier - limber) data vector"
      " at all tomographic bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("C_gg_tomo",
      py::overload_cast<arma::Col<double>>(&cosmolike_interface::C_gg_tomo_cpp),
      "Compute position-position (fourier - non-limber/limber) data vector"
      " at all tomographic bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("C_ks_tomo_limber",
      py::overload_cast<const double, const int>(
        &cosmolike_interface::C_ks_tomo_limber_cpp),
      "Compute CMB lensing-shear (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert()
    );

  m.def("C_ks_tomo_limber",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_ks_tomo_limber_cpp),
      "Compute CMB lensing-shear (fourier - limber) data vector at all"
      " tomographic bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // Derivative
  // --------------------------------------------------------------------

  m.def("dlnC_ss_dlnk_tomo_limber",
      py::overload_cast<const double, const double, const int, const int>(
        &cosmolike_interface::dlnC_ss_dlnk_tomo_limber_cpp
      ),
      "Compute dlnC_ss_dlnk (fourier - limber) derivative of the data vector",
      py::arg("k").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert(),
      py::return_value_policy::move
    );

  m.def("dlnC_ss_dlnk_tomo_limber",
      py::overload_cast<const arma::Col<double>, const arma::Col<double>>(
        &cosmolike_interface::dlnC_ss_dlnk_tomo_limber_cpp
      ),
      "Compute dlnC_ss_dlnk (fourier - limber) derivative of the data vector",
      py::arg("k").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("rf_C_ss_tomo_limber",
      py::overload_cast<const double, const double, const int, const int>(
        &cosmolike_interface::RF_C_ss_tomo_limber_cpp
      ),
      "Compute int from -infty to k of |dlnC_ss_dlnk| (fourier - limber)",
      py::arg("k").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert(),
      py::return_value_policy::move
    );

  m.def("rf_C_ss_tomo_limber",
      py::overload_cast<const arma::Col<double>, const arma::Col<double>>(
        &cosmolike_interface::RF_C_ss_tomo_limber_cpp
      ),
      "Compute int from -infty to k of |dlnC_ss_dlnk| (fourier - limber)",
      py::arg("k").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("dlnxi_dlnk_pm_tomo_limber",
      py::overload_cast<const double>(
        &cosmolike_interface::dlnxi_dlnk_pm_tomo_limber_cpp
      ),
      "Compute dlnxi_dlnk (real - limber) derivative of the data vector",
      py::arg("k").none(false),
      py::return_value_policy::move
    );

  m.def("dlnxi_dlnk_pm_tomo_limber",
      py::overload_cast<const arma::Col<double>>(
        &cosmolike_interface::dlnxi_dlnk_pm_tomo_limber_cpp
      ),
      "Compute dlnxi_dlnk (real - limber) derivative of the data vector",
      py::arg("k").none(false),
      py::return_value_policy::move
    );

  m.def("rf_xi_tomo_limber",
      py::overload_cast<const double, const int, const int, const int>(
        &cosmolike_interface::RF_xi_tomo_limber_cpp
      ),
      "Compute int from -infty to k of |dlnxi_dlnk| (fourier - limber)",
      py::arg("k").none(false),
      py::arg("nt").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert(),
      py::return_value_policy::move
    );
  
  m.def("rf_xi_tomo_limber",
      py::overload_cast<const arma::Col<double>>(
        &cosmolike_interface::RF_xi_tomo_limber_cpp
      ),
      "Compute int from -infty to k of |dlnxi_dlnk| (fourier - limber)",
      py::arg("k").none(false),
      py::return_value_policy::move
    );

  m.def("dlnC_ks_dlnk_tomo_limber",
      py::overload_cast<const double, const double, const int>(
        &cosmolike_interface::dlnC_ks_dlnk_tomo_limber_cpp
      ),
      "Compute dlnC_ks_dlnk (fourier - limber) derivative of the data vector",
      py::arg("k").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::return_value_policy::move
    );

  m.def("dlnC_ks_dlnk_tomo_limber",
      py::overload_cast<const arma::Col<double>, const arma::Col<double>>(
        &cosmolike_interface::dlnC_ks_dlnk_tomo_limber_cpp
      ),
      "Compute dlnC_ks_dlnk (fourier - limber) derivative of the data vector",
      py::arg("k").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("rf_C_ks_tomo_limber",
      py::overload_cast<const double, const double, const int>(
        &cosmolike_interface::RF_C_ks_tomo_limber_cpp
      ),
      "Compute int from -infty to k of |dlnC_ks_dlnk| (fourier - limber)",
      py::arg("k").none(false).noconvert(),
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::return_value_policy::move
    );

  m.def("rf_C_ks_tomo_limber",
      py::overload_cast<const arma::Col<double>, const arma::Col<double>>(
        &cosmolike_interface::RF_C_ks_tomo_limber_cpp
      ),
      "Compute int from -infty to k of |dlnC_ks_dlnk| (fourier - limber)",
      py::arg("k").none(false),
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("dlnw_ks_dlnk_tomo",
      py::overload_cast<const double>(
        &cosmolike_interface::dlnw_ks_dlnk_tomo_cpp
      ),
      "Compute dlnw_ks_dlnk (real - limber) derivative of the data vector",
      py::arg("k").none(false),
      py::return_value_policy::move
    );

  m.def("dlnw_ks_dlnk_tomo",
      py::overload_cast<const arma::Col<double>>(
        &cosmolike_interface::dlnw_ks_dlnk_tomo_cpp
      ),
      "Compute dlnw_ks_dlnk (real - limber) derivative of the data vector",
      py::arg("k").none(false),
      py::return_value_policy::move
    );

  m.def("rf_w_ks_tomo",
      py::overload_cast<const double, const int, const int>(
        &cosmolike_interface::RF_w_ks_tomo_cpp
      ),
      "Compute int from -infty to k of |dlnw_ks_dlnk| (fourier - limber)",
      py::arg("k").none(false),
      py::arg("nt").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::return_value_policy::move
    );

  m.def("rf_w_ks_tomo",
      py::overload_cast<const arma::Col<double>>(
        &cosmolike_interface::RF_w_ks_tomo_cpp
      ),
      "Compute int from -infty to k of |dlnw_ks_dlnk| (fourier - limber)",
      py::arg("k").none(false),
      py::return_value_policy::move
    );

  // --------------------------------------------------------------------
  // Halo model (halo.c)
  // --------------------------------------------------------------------
  m.def("hb1nu",
      &cosmolike_interface::hb1nu_cpp,
      "Tinker et al. 2010 halo bias b(nu) at peak height "
      "nu = delta_c/sigma(M, a)",
      py::arg("nu").none(false),
      py::arg("a").none(false)
    );

  m.def("fnu",
      &cosmolike_interface::fnu_cpp,
      "Tinker et al. 2010 multiplicity function f(nu) of the halo mass "
      "function at peak height nu (0 < a < 1)",
      py::arg("nu").none(false),
      py::arg("a").none(false)
    );

  m.def("conc",
      &cosmolike_interface::conc_cpp,
      "Halo concentration c(m) (Bhattacharya et al. 2013, Delta = 200 "
      "mean); m in M_sun/h, growfac_a = D(a)",
      py::arg("m").none(false),
      py::arg("growfac_a").none(false)
    );

  m.def("dlognudlogm",
      &cosmolike_interface::dlognudlogm_cpp,
      "Slope dln nu/dln M of the peak height (cached table at a = 1); "
      "M in M_sun/h",
      py::arg("M").none(false)
    );

  m.def("bias_norm",
      &cosmolike_interface::bias_norm_cpp,
      "Halo-bias normalization int b(nu) f(nu) dnu over the tabulated "
      "mass range (cached table in a)",
      py::arg("a").none(false)
    );

  m.def("u_nfw_c",
      &cosmolike_interface::u_nfw_c_cpp,
      "Fourier transform of the NFW profile, normalized to 1 at k = 0; "
      "k in (c/H0)^-1, m in M_sun/h",
      py::arg("c").none(false),
      py::arg("k").none(false),
      py::arg("m").none(false),
      py::arg("a").none(false)
    );

  m.def("u_KS",
      &cosmolike_interface::u_KS_cpp,
      "Fourier transform of the Komatsu-Seljak gas pressure profile "
      "(cached table); k in (c/H0)^-1, rv in c/H0",
      py::arg("c").none(false),
      py::arg("k").none(false),
      py::arg("rv").none(false)
    );

  m.def("ngal",
      &cosmolike_interface::ngal_cpp,
      "HOD galaxy number density of lens bin ni in (c/H0)^-3 (cached "
      "table)",
      py::arg("ni").none(false).noconvert(),
      py::arg("a").none(false)
    );

  m.def("bgal",
      &cosmolike_interface::bgal_cpp,
      "HOD bias-weighted integral of lens bin ni (cached table)",
      py::arg("ni").none(false).noconvert(),
      py::arg("a").none(false)
    );

  m.def("p_mm",
      py::overload_cast<const double, const double>(
        &cosmolike_interface::p_mm_cpp
      ),
      "Halo-model matter power spectrum at one (k, a); k in (c/H0)^-1, "
      "P in (c/H0)^3",
      py::arg("k").none(false).noconvert(),
      py::arg("a").none(false).noconvert()
    );

  m.def("p_mm",
      py::overload_cast<const arma::Col<double>, const double>(
        &cosmolike_interface::p_mm_cpp
      ),
      "Halo-model matter power spectrum at many k, one a (vectorized)",
      py::arg("k").none(false),
      py::arg("a").none(false),
      py::return_value_policy::move
    );

  m.def("p_gm",
      py::overload_cast<const double, const double, const int>(
        &cosmolike_interface::p_gm_cpp
      ),
      "Halo-model galaxy-matter power spectrum of lens bin ni at one "
      "(k, a); k in (c/H0)^-1, P in (c/H0)^3",
      py::arg("k").none(false).noconvert(),
      py::arg("a").none(false).noconvert(),
      py::arg("ni").none(false).noconvert()
    );

  m.def("p_gm",
      py::overload_cast<const arma::Col<double>, const double, const int>(
        &cosmolike_interface::p_gm_cpp
      ),
      "Halo-model galaxy-matter power spectrum of lens bin ni at many k, "
      "one a (vectorized)",
      py::arg("k").none(false),
      py::arg("a").none(false),
      py::arg("ni").none(false).noconvert(),
      py::return_value_policy::move
    );

  m.def("p_gg",
      py::overload_cast<const double, const double, const int, const int>(
        &cosmolike_interface::p_gg_cpp
      ),
      "Halo-model galaxy-galaxy power spectrum of lens bin ni (nj = ni) "
      "at one (k, a); k in (c/H0)^-1, P in (c/H0)^3",
      py::arg("k").none(false).noconvert(),
      py::arg("a").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert()
    );

  m.def("p_gg",
      py::overload_cast<const arma::Col<double>, const double,
                        const int, const int>(
        &cosmolike_interface::p_gg_cpp
      ),
      "Halo-model galaxy-galaxy power spectrum of lens bin ni (nj = ni) "
      "at many k, one a (vectorized)",
      py::arg("k").none(false),
      py::arg("a").none(false),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert(),
      py::return_value_policy::move
    );

  m.def("ia_f_red_central",
      &cosmolike_interface::ia_f_red_central_cpp,
      "Halo-model IA red-central fraction f_rc(a) of the source sample "
      "(cached table; 0 outside the source a range)",
      py::arg("a").none(false)
    );

  m.def("ia_window_2h",
      py::overload_cast<const double>(
        &cosmolike_interface::ia_window_2h_cpp
      ),
      "Halo-model IA window of the NLA 2-halo term, f_2h(k) = "
      "exp[-(k/k_2h)^2], at one k; k in (c/H0)^-1",
      py::arg("k").none(false).noconvert()
    );

  m.def("ia_window_2h",
      py::overload_cast<const arma::Col<double>>(
        &cosmolike_interface::ia_window_2h_cpp
      ),
      "Halo-model IA window of the NLA 2-halo term at many k (vectorized)",
      py::arg("k").none(false),
      py::return_value_policy::move
    );

  m.def("ia_p1h_dI",
      py::overload_cast<const double, const double>(
        &cosmolike_interface::ia_p1h_dI_cpp
      ),
      "Halo-model IA 1-halo matter-intrinsic spectrum at one (k, a), "
      "signed with a_1h (the C_l cores subtract it); k in (c/H0)^-1, "
      "P in (c/H0)^3",
      py::arg("k").none(false).noconvert(),
      py::arg("a").none(false).noconvert()
    );

  m.def("ia_p1h_dI",
      py::overload_cast<const arma::Col<double>, const double>(
        &cosmolike_interface::ia_p1h_dI_cpp
      ),
      "Halo-model IA 1-halo matter-intrinsic spectrum at many k, one a "
      "(vectorized)",
      py::arg("k").none(false),
      py::arg("a").none(false),
      py::return_value_policy::move
    );

  m.def("ia_p1h_II",
      py::overload_cast<const double, const double>(
        &cosmolike_interface::ia_p1h_II_cpp
      ),
      "Halo-model IA 1-halo intrinsic-intrinsic spectrum at one (k, a); "
      "k in (c/H0)^-1, P in (c/H0)^3",
      py::arg("k").none(false).noconvert(),
      py::arg("a").none(false).noconvert()
    );

  m.def("ia_p1h_II",
      py::overload_cast<const arma::Col<double>, const double>(
        &cosmolike_interface::ia_p1h_II_cpp
      ),
      "Halo-model IA 1-halo intrinsic-intrinsic spectrum at many k, one a "
      "(vectorized)",
      py::arg("k").none(false),
      py::arg("a").none(false),
      py::return_value_policy::move
    );

  m.def("growfac",
      &cosmolike_interface::growfac_cpp,
      "Linear growth factor D(a), D(1) = 1 (halo-model input)",
      py::arg("a").none(false)
    );

  m.def("p_lin",
      &cosmolike_interface::p_lin_cpp,
      "Linear matter power spectrum at one (k, a); k in (c/H0)^-1, P in "
      "(c/H0)^3 (halo-model input)",
      py::arg("k").none(false),
      py::arg("a").none(false)
    );

  m.def("Pdelta",
      &cosmolike_interface::Pdelta_cpp,
      "Run-mode (nonlinear) matter power spectrum at one (k, a); k in "
      "(c/H0)^-1, P in (c/H0)^3 (halo-model input)",
      py::arg("k").none(false),
      py::arg("a").none(false)
    );

  m.def("set_HOD",
      &cosmolike_interface::set_HOD_cpp,
      "Load halo.c's built-in Coupon et al. 2012 HOD for lens bin ni",
      py::arg("ni").none(false).noconvert()
    );

  m.def("set_nuisance_hod",
      &cosmolike_interface::set_nuisance_hod_cpp,
      "Set the HOD {lgMmin, sigma_lgM, lgM1, lgM0, alpha, f_c} and the "
      "galaxy concentration factor gc of lens bin ni",
      py::arg("ni").none(false).noconvert(),
      py::arg("hod").none(false),
      py::arg("gc").none(false)
    );

  m.def("set_nuisance_gas",
      &cosmolike_interface::set_nuisance_gas_cpp,
      "Set the gas (Compton-y) parameters nuisance.gas[0..n-1]",
      py::arg("gas").none(false)
    );

  m.def("set_nuisance_ia_halo",
      &cosmolike_interface::set_nuisance_ia_halo_cpp,
      "Set the halo-model IA parameters: ia_halo = {a_1h, eta_1h, "
      "z_pivot}, ia_red = the four red-fraction sigmoid parameters, "
      "ia_hod = the six IA-population HOD parameters",
      py::arg("ia_halo").none(false),
      py::arg("ia_red").none(false),
      py::arg("ia_hod").none(false)
    );

  // --------------------------------------------------------------------
  // Miscellaneous
  // --------------------------------------------------------------------
  m.def("get_mask",
      // Why return an STL vector?
      // The conversion between STL vector and python np array is cleaner
      // arma:Col is cast to 2D np array with 1 column (not as nice!)
      []()->std::vector<int>{
        using namespace cosmolike_interface;
        arma::Col<int> res = IP::get_instance().get_mask();
        return arma::conv_to<std::vector<int>>::from(res);
      },
      "Get Mask Vector",
      py::return_value_policy::move
    );

  m.def("get_dv_masked",
      // Why return an STL vector?
      // The conversion between STL vector and python np array is cleaner
      // arma:Col is cast to 2D np array with 1 column (not as nice!)
      []()->std::vector<double> {
        using namespace cosmolike_interface;
        arma::Col<double> res = IP::get_instance().get_dv_masked();
        return arma::conv_to<std::vector<double>>::from(res);
      },
      "Get Mask Data Vector",
      py::return_value_policy::move
    );

  m.def("get_cov_masked",
      []()->arma::Mat<double> {
        return cosmolike_interface::IP::get_instance().get_cov_masked();
      },
      "Get Mask Covariance Matrix",
      py::return_value_policy::move
    );

  m.def("get_inv_cov_masked",
      []()->arma::Mat<double> {
        return cosmolike_interface::IP::get_instance().get_inv_cov_masked();
      },
      "Get Mask Covariance Matrix",
      py::return_value_policy::move
    );
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int main()
{
  std::cout << "GOODBYE" << std::endl;
  exit(1);
}
