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
