# desy1xplanck data

This folder holds the DES Y3 × Planck PR4 6x2pt data set: the measured data vector, its covariance, the scale-cut masks, the redshift distributions, the baryon files, and the CMB-lensing inputs. Every likelihood of the project (the `data_file` key of `../likelihood/*.yaml`) and every example loads it through the descriptor `Y3xPlanckPR4.dataset`. The data vector is a measurement, combining DES Y3 galaxy shapes and MagLim lens galaxies with the Planck PR4 map of CMB lensing; it is not a synthetic vector. [likelihood/README.md](../likelihood/README.md#desy1xplanck_mask) gives the order of its 1,809 entries and, per block, the entries the baseline mask keeps.

| file | content |
|---|---|
| `Y3xPlanckPR4.dataset` | the descriptor: the file names below, six lens and four source bins, 30 angular bins between 0.25 and 250 arcmin, the filter of the CMB-lensing map in the cross-correlations (beam `fwhm_kx` = 7 arcmin, multipoles `lmin_kx` = 8 to `lmax_kx` = 2048, the HEALPix pixel window), and the CMB-lensing bandpowers (`nbp_kk` = 9 bands, the binning matrix and offset of `cmb_data/dr4_consext8_CMBmarged`, and `hartlap_nvar_kk` = 480 simulations behind their covariance) |
| `DESY3xPlanckPR4_6x2pt_Maglim_baseline_archive.realdata` | the measured data vector, 1,809 entries (columns: index, value) |
| `DESY3xPlanckPR4_6x2pt_baseline_archive.covmat` | its covariance, a text table of the full 1,809 × 1,809 matrix (columns: i, j, value; 3,272,481 rows, 71 MB). Stored with Git LFS |
| `DESY3xPlanckR4_6x2pt_Maglim_baseline_archive.mask` | the baseline scale-cut mask (columns: index, 0 or 1); it keeps 738 of the 1,809 entries |
| `ones.mask` | a mask that keeps all 1,809 entries |
| `nz_maglim_Y3_unblinded_02_26_21.txt` | redshift distributions of the six MagLim lens bins (columns: z, then one column per bin) |
| `nz_source_Y3_unblinded_02_26_21.txt` | redshift distributions of the four source bins (columns: z, then one column per bin) |
| `nz_redmagic_Y3_unblinded_02_24_21.txt` | redshift distributions of the five redMaGiC lens bins; no dataset of the project names this file |
| `DESY3xPLKR4_6x2pt_antilles_baseline_archive.pca` | baryonic-feedback principal components of the data vector, one row per entry and one column per component (six columns). The likelihoods read it only with `use_baryon_pca: True`, and then sample the amplitudes `DES_BARYON_Q1` ... `DES_BARYON_Q4` |
| `baryons_logPkR.h5` | the ratios $P_{\rm hydro}(k,z)/P_{\rm DMO}(k,z)$ of hydrodynamical simulations, read only by a run that builds the baryon principal components (`create_baryon_pca: True`) or contaminates the data vector with one simulation (`add_baryons_on_dv: True`). Stored with Git LFS |
| `cmb_data/` | the CMB-lensing inputs of the next table |

## CMB-lensing inputs (`cmb_data/`)

| file or folder | content |
|---|---|
| `HPIX1024_pixwin.txt` | the HEALPix pixel window at $N_{\rm side} = 1024$ (columns: multipole, window; multipoles 0 to 3071), the `healpix_win_func_kx_file` of the descriptor |
| `dr4_consext8_CMBmarged/` | the bandpowers of the CMB-lensing auto-spectrum that the descriptor reads: nine bands over multipoles 8 to 400, the binning matrix with the primary-CMB corrections (`binmat_kk_file`), and the bandpower offset (`offset_kk_file`) |
| `pp_agr2_CMBmarged/`, `pp_consext8_CMBmarged/` | other bandpower sets of the Planck lensing likelihood for the minimum-variance estimator (`pp_agr2`: 14 bands over multipoles 8 to 2048); no dataset of the project names them |
| `pttptt_agr2_CMBmarged/` | a bandpower set for the temperature-only estimator (covariance, data vector, offset, and two binning matrices); no dataset of the project names it |
| `noise_plancksmica.txt` | a two-column noise table (101 rows) that no file of the project reads |

The folders `dr4_consext8_CMBmarged`, `pp_agr2_CMBmarged`, and `pp_consext8_CMBmarged` each have a README that describes their files.

## Git LFS

The covariance `DESY3xPlanckPR4_6x2pt_baseline_archive.covmat` and `baryons_logPkR.h5` are stored with Git LFS (the patterns are in `../.gitattributes`). A clone without `git lfs pull` holds a pointer file in place of each, and the likelihoods cannot load it.

## The tests' copy

The unit tests keep their own copy of the data set under `../tests/frozen/data`, protected by `../tests/manifest_sha256.json`, so a change to a file of this folder does not reach them until the snapshot is refreshed (see [tests/data_vector/README.md](../tests/data_vector/README.md#frozen_copy)).
