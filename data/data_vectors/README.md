
### Synthetic Data

desy1_theory_LCDM.modelvector
desy1_theory_LCDM_mnu0.25.modelvector
desy1_theory_LCDM_mnu0.25_illustris.modelvector
desy1_theory_LCDM_ssExt_025.modelvector
desy1_theory_LCDM_ssExt_050.modelvector
desy1_theory_LCDM_ssExt_100.modelvector
desy1xplanck_6x2pt_theory_LCDM_dmo.modelvector
desy1xplanck_6x2pt_theory_LCDM_eagle.modelvector
desy1xplanck_6x2pt_theory_LCDM_illustris.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_025_dmo.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_025_dmo_highres.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_025_eagle.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_025_eagle_highres.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_025_illustris.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_025_illustris_highres.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_050_dmo.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_050_dmo_highres.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_050_eagle.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_050_eagle_highres.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_050_illustris.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_050_illustris_highres.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_100_dmo.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_100_dmo_highres.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_100_eagle.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_100_eagle_highres.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_100_illustris.modelvector
desy1xplanck_6x2pt_theory_LCDM_ssExt_100_illustris_highres.modelvector
desy3_theory_LCDM.modelvector

- `desy1xplanck_6x2pt_theory_LCDM_ssExt_???_dmo/eagle/illustris_agr2_highres.modelvector`: synthetic data vector with different baryon contamination scenarios, with different `theta_min`, and with L in [8, 2048]. The computation accuracy is high such that small-scales `xi_pm` can be calculated accurately. 

- `desy1xplanck_6x2pt_theory_LCDM_eagle_consext8.modelvector`: using the pr3 consext8 setting, including L-cut

- `desy1xplanck_6x2pt_theory_LCDM_eagle_pr4.modelvector`: using the pr3 consext8 + pr4 kk setting, including L-cut

### Real Data

Files end with `_Reference/tSZ-contam/tSZ-deproj` have L-cut consistent with
the CMB lensing L-cut description, e.g. `agr2`=[8, 2048], `consext8`=[8,400]. 
Otherwise, the L-cut is [40, 2999]

*Planck PR4*: 3x2pt measured from DES Y1, 5x2pt measured from DES Y1 x PR3 MV. Ckk from PR4. tSZ treatment are stated in filenames.
- `xi_desy1xplanck_6x2pt_realdata_dr4_consext8_CMBmarged`: Y1 L-cut
- `xi_desy1xplanck_6x2pt_realdata_dr4_consext8_CMBmarged_Reference`: consext8 L-cut
- `xi_desy1xplanck_6x2pt_realdata_dr4_consext8_CMBmarged_tSZ-contam`: consext8 L-cut
- `xi_desy1xplanck_6x2pt_realdata_dr4_consext8_CMBmarged_tSZ-deproj`: consext8 L-cut
- `xi_desy1xplanck_6x2pt_realdata_dr4_consext8_CMBmarged_ssExt`: Y1 L-cut
- `xi_desy1xplanck_6x2pt_realdata_dr4_consext8_CMBmarged_ssExt_Reference`: consext8 L-cut
- `xi_desy1xplanck_6x2pt_realdata_dr4_consext8_CMBmarged_ssExt_tSZ-contam`: consext8 L-cut
- `xi_desy1xplanck_6x2pt_realdata_dr4_consext8_CMBmarged_ssExt_tSZ-deproj`: consext8 L-cut

consext8 version

without extended annulus

dr4 

*Planck PR3*: 3x2pt measured from DES Y1, 5x2pt measured from DES Y1 x PR3 MV. Ckk from PR3. tSZ treatment stated in filenames.
- `xi_desy1xplanck_6x2pt_realdata_pp_agr2_CMBmarged`: Y1 L-cut [40, 2999]
- `xi_desy1xplanck_6x2pt_realdata_pp_agr2_CMBmarged_Reference`: agr2 L-cut
- `xi_desy1xplanck_6x2pt_realdata_pp_agr2_CMBmarged_tSZ-contam`: agr2 L-cut
- `xi_desy1xplanck_6x2pt_realdata_pp_agr2_CMBmarged_tSZ-deproj`: agr2 L-cut
- `xi_desy1xplanck_6x2pt_realdata_pp_agr2_CMBmarged_ssExt`: Y1 L-cut [40, 2999]
- `xi_desy1xplanck_6x2pt_realdata_pp_agr2_CMBmarged_ssExt_Reference`: agr2 L-cut
- `xi_desy1xplanck_6x2pt_realdata_pp_agr2_CMBmarged_ssExt_tSZ-contam`: agr2 L-cut
- `xi_desy1xplanck_6x2pt_realdata_pp_agr2_CMBmarged_ssExt_tSZ-deproj`: agr2 L-cut

*DES Y3*:
- `xi_desy3xplanck_6x2pt_realdata_pp_agr2_CMBmarged`: Y3 with L in [40, 299]

*Legacy*: the following dvs are not used.
xi_desy1xplanck_6x2pt_simudata_dr4_consext8_CMBmarged_cocoa
xi_desy1xplanck_6x2pt_simudata_dr4_consext8_CMBmarged_cosmolike
xi_desy1xplanck_6x2pt_simudata_pp_agr2_CMBmarged_cocoa
xi_desy1xplanck_6x2pt_simudata_pp_agr2_CMBmarged_cosmolike

**Actively Using**
- `xi_desy1xplanck_6x2pt_realdata_20_wA_ref_pp_agr2`
- `xi_desy1xplanck_6x2pt_realdata_20_wA_ref_pp_consext8`
- `xi_desy1xplanck_6x2pt_realdata_20_wA_ref_pp_pr4`
- `xi_desy1xplanck_6x2pt_realdata_20_wA_tSZc_pp_agr2`
- `xi_desy1xplanck_6x2pt_realdata_20_wA_tSZc_pp_consext8`
- `xi_desy1xplanck_6x2pt_realdata_20_wA_tSZc_pp_pr4`
- `xi_desy1xplanck_6x2pt_realdata_20_wA_tSZd_pp_agr2`
- `xi_desy1xplanck_6x2pt_realdata_20_wA_tSZd_pp_consext8`
- `xi_desy1xplanck_6x2pt_realdata_20_wA_tSZd_pp_pr4`
- `xi_desy1xplanck_6x2pt_realdata_20_woA_ref_pp_agr2`
- `xi_desy1xplanck_6x2pt_realdata_20_woA_ref_pp_consext8`
- `xi_desy1xplanck_6x2pt_realdata_20_woA_ref_pp_pr4`
- `xi_desy1xplanck_6x2pt_realdata_30_wA_ref_pp_agr2`
- `xi_desy1xplanck_6x2pt_realdata_30_wA_ref_pp_consext8`
- `xi_desy1xplanck_6x2pt_realdata_30_wA_ref_pp_pr4`
