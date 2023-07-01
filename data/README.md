# Fiducial Parameters of `xi_desy1xplanck_6x2pt_fid_cosmolike_core`

## Cosmological Parameters

- `omega_m` = $\Omega\_m$ = 0.3
- `sigma_8` = $\sigma\_8$ = 0.827726
- `n_s` = $n\_s$ = 0.96605
- `w0` = $w\_0$ = -1.0
- `wa` = $w\_a$ = 0.0
- `omega_b` = $\Omega\_b$ = 0.04
- `omega_nuh2` = $\Omega\_vh^2$ = 0.06/93.14
- `h0` = $h$ = 0.6732
- `MGSigma` = $Sigma$ = 0.0
- `MGmu` = $\mu$ = 0.0

`cosmolike_core` code example:
> input\_cosmo\_params\_y3 ic = {
>        .omega_m = 0.3,
>        .sigma_8 = 0.827726,
>        //.A_s = 2.1e-9,
>        .n_s = 0.96605,
>        .w0 = -1.0,
>        .wa = 0.0,
>        .omega_b = 0.04,
>        .omega_nuh2 = 0.06/93.14,
>        .h0 = 0.6732,
>        .MGSigma = 0.0,
>        .MGmu = 0.0,
>        //.theta_s = -1.0,
>    };

## Nuisance Parameters

- `bias` = $b\_1$ = [1.72716, 1.65168, 1.61423, 1.92886, 2.11633, 0.0, 0.0, 0.0, 0.0, 0.0]
- `b2` = $b\_2$ = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- `b_mag` = $b\_m$ = [-0.19375, -0.6285407, -0.69319886, 1.17735723, 1.87509758,  0.0, 0.0, 0.0, 0.0, 0.0]
- `lens_z_bias` = $\Delta z\_{lens}$ = [0.00457604, 0.000309875, 0.00855907, -0.00316269, -0.0146753, 0.0, 0.0, 0.0, 0.0, 0.0]
- `source_z_bias` = $\Delta z\_{src}$ = [0.0414632, 0.00147332, 0.0237035, -0.0773436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- `shear_m` = $m$ = [0.0191832, -0.0431752, -0.034961, -0.0158096, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- `p_ia` = [0.606102, -1.51541, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

`cosmolike_core` code example:
input_nuisance_params_y3 in = {
>        .bias = {1.72716, 1.65168, 1.61423, 1.92886, 2.11633,
>                0.0, 0.0, 0.0, 0.0, 0.0},
>        .b_mag = {-0.19375, -0.6285407, -0.69319886, 1.17735723, 1.87509758,
>                0.0, 0.0, 0.0, 0.0, 0.0},
>        .lens_z_bias = {0.00457604, 0.000309875, 0.00855907, -0.00316269,
>            -0.0146753, 0.0, 0.0, 0.0, 0.0, 0.0},
>        .source_z_bias = {0.0414632, 0.00147332, 0.0237035, -0.0773436,
>            0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
>        .shear_m = {0.0191832, -0.0431752, -0.034961, -0.0158096,
>            0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
>        .p_ia = {0.606102, -1.51541, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
>    };

## Other Settings

`cosmolike_core` code example:

>    double theta_min = 2.5, theta_max = 250.0;
>    int Ntheta = 20;
>    double l_min = 40.0, l_max = 3000.0;
>    int Ncl = 15;
>    double Rmin_bias = 20.0;
>    double lmax_shear = 3000.0;
>    double lmax_kappacmb = 2999.0;
>    double ggl_cut = 0.0;
>    int ntomo_source = 4, ntomo_lens = 5;
>    char runmode[50] = "Halofit";
>    char source_nz[500] = "./zdistris/mcal_1101_source.nz";
>    char lens_nz[500] = "./zdistris/mcal_1101_lens.nz";
>    char probes[50] = "6x2pt";
>    char cmbName[50] = "planck";
>    char cmb_lens_noise_file[500] = "./cmblensrec/plancksmica/cmb_lmax3000.txt";
>    char cov_file[500] = "./covs/cov_y1xplanck_mix6x2pt_referebceMV_new";
>    char data_file[500] = "./datav/xi_desy1xplanck_6x2pt_fid";
>    char mask_file[500] = "./yaml/xi_desy1xplanck_6x2pt_fid.mask";
>    char test_model_file[500] = "./datav/xi_desy1xplanck_6x2pt_mv_at_fiducial";
>    int IA_model = 4; // 4 = NLA

## File Description

`cov_y1xplanck_mix6x2pt_pp_p18cosmo_kk_sim`: mix 6x2pt covariance matrix, with CMB lensing in band power. The CMB lensing auto-covariance is from Planck 2018 data release

`pp_agr2_CMBmarged`: data files of Planck lensing likelihood, assumes minimum-variance estimator

`pttptt_agr2_CMBmarged`: data files of Planck lensing likelihood, assumes TT-only estimator

`xi_desy1xplanck_6x2pt_bp.mask`: mask file, assume 14 CMB lensing band power bins

## Dataset file description
Here we clarify the details of the `*.dataset` files

- `Y1xplanck.dataset`:
- `Y1xplanck_dr4_consext8_CMBmarg.dataset`:
- `Y3xplanck.dataset`:
- `Y6xplanck.dataset`:

*Real Datavector*: These dataset files are using real measured data vectors

_Different scale-cuts:_

- `Y1xplanck_Y1cut_real.dataset`: Y1 standard scale-cuts (including `xi_pm`, L in [8, 2048]). Remember to manually check what datavector is used before running
- `Y1xplanck_100_real.dataset`: Y1 standard scale-cuts except `xi_pm`, which is 1.0 arcmin. Now using L in [8, 2048]. Remember to manually check before running chains.
- `Y1xplanck_250_real.dataset`: same as above, but 2.5 arcmin scale cut on `xi_pm`. Now using agr2 L-cut, but remember to manually check before running chains

_Different tSZ recipes:_

- `Y1xplanck_250_real_tSZcontam.dataset`: same as above, but measured using tSZ-contaminated Planck lensing map (L in [8, 2048], agr2)
- `Y1xplanck_250_real_tSZdeproj.dataset`: same as above, but measured using tSZ-deprojected Planck lensing map (L in [8, 2048], agr2)

_Not used:_

- `Y1xplanck_ssExt_025.dataset`:  real data measured with agr2 lensing map (L in [8, 2048]), `xi_pm` cut at 0.25 arcmin
- `Y1xplanck_ssExt_050.dataset`: same as above, but cut at 0.5 arcmin
- `Y1xplanck_ssExt_100.dataset`: same as above, but cut at 1.0 arcmin, now it's a duplicate of `Y1xplanck_100_real.dataset`. 

*Synthetic Datavector for baryon effect scale-cut analyses*: These dataset files are using synthetic data vectors with dmo/illustris/eagle baryon contamination scenarios, with different `xi_pm` scale-cuts

- `Y1xplanck_dmo.dataset`: the standard `Ntheta=20` binning with DMO scenario. Y1 cut but with `theta_min=2.5 arcmin` for `xi_pm`.
- `Y1xplanck_eagle.dataset`: sab (same as above) but contaminated by EAGLE 
- `Y1xplanck_illustris.dataset`" sab but contaminated by Illustris
- `Y1xplanck_Y1cut_dmo.dataset`: Y1 standard scale-cuts with DMO.
- `Y1xplanck_Y1cut_eagle.dataset`: same as above, but contaminated by EAGLE
- `Y1xplanck_Y1cut_illustris.dataset`: same as above, but contaminated by Illustris
- `Y1xplanck_ssExt_025_dmo.dataset`: sab but has `xi_pm` measurement extended to 0.25 arcmin
- `Y1xplanck_ssExt_025_eagle.dataset`: can be inferred from above
- `Y1xplanck_ssExt_025_illustris.dataset`: can be inferred from above
- `Y1xplanck_ssExt_050_dmo.dataset`: can be inferred from above
- `Y1xplanck_ssExt_050_eagle.dataset`: can be inferred from above
- `Y1xplanck_ssExt_050_illustris.dataset`:can be inferred from above
- `Y1xplanck_ssExt_100_dmo.dataset`: can be inferred from above
- `Y1xplanck_ssExt_100_eagle.dataset`: can be inferred from above
- `Y1xplanck_ssExt_100_illustris.dataset`: can be inferred from above

*Misc*: 

- `Y1xplanck_5x2pt+C_kk.dataset`: Dataset to test the impact of cross-cov between 5x2pt and Ckk. It uses a covmat with cross-cov terms set to zero.
- `Y1xplanck_withoutAnnulus.dataset`: Dataset to test the impact of survey-geometry correction. It uses a covmat assuming the `w_gk` and `w_sk` measurements are using a Planck lensing map cutted to the Y1 footprint. 
