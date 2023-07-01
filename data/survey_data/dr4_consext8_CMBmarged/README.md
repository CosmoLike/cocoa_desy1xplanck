### NOTE: the covariance has Ckk-Ckk block from Planck 2018 covariance, and the Hartlap factor has to be applied to the inverse covariance when calculating the likelihood. The covariance matrix itself doesn't include that factor!!!

## Files

- `Ckk_bandpower_covariance.txt`: covariance matrix of Ckk reduced from FFP10 simulations. Hartlap factor has to be applied
- `Ckk_bandpower_datavector.txt`: Ckk band power
- `Ckk_bandpower_offset.txt`: Ckk band power offset due to marginalization over primary CMB
- `binning_matrix_table.txt`: $\mathcal{B}_i^L$, used to calculate cross-cov between 5x2pt and Ckk. Has 9 rows (bin 1 - 9) and 392 columns (L in [8, 400])
- `binning_matrix_table_extended.txt`: same as `binning_matrix_table.txt` but embedded to shape 9 rows x 2499 (L in [2, 2500])
- `binning_matrix_with_correction_table.txt`: $\mathrm{B}_i^L + M_i^{\kappa, L}$ with corrections for primary-CMB dependency, used to calculate cross-cov between 5x2pt and Ckk. Has 9 rows (bin 1 - 9) and 2499 columns (L in [2, 2500])
