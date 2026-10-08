# Likelihoods and their parameters <a name="desy1xplanck_likelihood_readme"></a>

This folder holds the likelihoods of the project (one yaml file with the defaults and one python file per likelihood), the parameter files the likelihoods include, and the base class `_cosmolike_prototype_base.py`.

1. [The likelihoods](#desy1xplanck_likelihoods)
2. [The parameter files and `fixed_params`](#desy1xplanck_parameter_files)
3. [The mask and lens bins 5-6](#desy1xplanck_mask)
4. [What each combination fixes](#desy1xplanck_fixed_params)
5. [Parameters that should not be varied](#desy1xplanck_not_varied)
6. [Changing the mask, the scale cuts, or the probes](#desy1xplanck_changing)

# The likelihoods <a name="desy1xplanck_likelihoods"></a>

Every likelihood reads the same data vector, covariance, and mask through `Y3xPlanckPR4.dataset`, and computes only the blocks it holds; the entries of the other blocks are zero in the theory vector and removed from the covariance.

| likelihood | blocks | parameter files |
|---|---|---|
| `cosmic_shear` | `ss` | `params_source.yaml` |
| `combo_2x2pt_ss_sk` | `ss`, `ks` | `params_source.yaml` |
| `combo_3x2pt_ss_sk_sk` | `ss`, `ks`, `kk` | `params_source.yaml` |
| `combo_2x2pt` | `gs`, `gg` | `params_source.yaml`, `params_lens_maglim.yaml` |
| `combo_2x2pt_ss_sg` | `ss`, `gs` | `params_source.yaml`, `params_lens_maglim.yaml` |
| `combo_3x2pt` | `ss`, `gs`, `gg` | `params_source.yaml`, `params_lens_maglim.yaml` |
| `combo_3x2pt_ks_gk_kk` | `gk`, `ks`, `kk` | `params_source.yaml`, `params_lens_maglim.yaml` |
| `combo_6x2pt` | `ss`, `gs`, `gg`, `gk`, `ks`, `kk` | `params_source.yaml`, `params_lens_maglim.yaml` |

The blocks:

| block | content |
|---|---|
| `ss` | cosmic shear $\xi_+$ and $\xi_-$ |
| `gs` | galaxy-galaxy lensing $\gamma_t$ |
| `gg` | galaxy clustering $w(\theta)$ |
| `gk` | galaxy density x CMB lensing |
| `ks` | galaxy shear x CMB lensing |
| `kk` | CMB lensing auto-spectrum bandpowers |

# The parameter files and `fixed_params` <a name="desy1xplanck_parameter_files"></a>

There is one parameter file per galaxy sample, shared by every likelihood that uses the sample:

| file | sample | parameters |
|---|---|---|
| `params_source.yaml` | DES Y3 sources, 4 bins | photo-z shift `DES_DZ_S1` ... `DES_DZ_S4`, shear calibration `DES_M1` ... `DES_M4`, intrinsic alignment `DES_A1_*`, `DES_A2_*`, `DES_BTA_*`, baryon PC amplitudes `DES_BARYON_Q1` ... `DES_BARYON_Q4` |
| `params_lens_maglim.yaml` | MagLim lenses, 6 bins | photo-z shift `DES_DZ_L*`, photo-z stretch `DES_DZ2_L*`, linear bias `DES_B1_*`, nonlinear bias `DES_B2_*`, magnification `DES_BMAG_*`, point mass `DES_PM*` |
| `params_lens_redmagic.yaml` | redMaGiC lenses, 5 bins | the same families; no likelihood of this folder includes it |

Each likelihood yaml includes its parameter files with cobaya's `!defaults` tag:

    [Adapted from likelihood/combo_6x2pt.yaml]
    params: !defaults [params_source, params_lens_maglim]

The tag builds the whole `params` mapping from the files, so the likelihood yaml cannot change one of its entries. A combination that fixes some of these parameters lists them in its `fixed_params` block instead:

    [Adapted from likelihood/combo_6x2pt.yaml]
    fixed_params:
      DES_DZ_L5:
        value: 0.002
      (...)
      DES_PM6:
        value: 0.0

The base class applies this block to the defaults before cobaya merges them with the user's yaml. Each entry replaces the entry of the parameter file with cobaya's merge rule: a `value` drops the `prior`, the `ref`, and the `proposal`, and the `latex` label stays. The parameter stays declared, as a constant. The combinations therefore share the parameter files and differ only in their `fixed_params`.

# The mask and lens bins 5-6 <a name="desy1xplanck_mask"></a>

`Y3xPlanckPR4.dataset` names the mask `DESY3xPlanckR4_6x2pt_Maglim_baseline_archive.mask`, which keeps an entry of the data vector when its second column is 1. The data vector has 1809 entries, in this order:

    ss  entries    0 -  599   xi+ then xi-, 10 source pairs (i <= j) x 30 angles each
    gs  entries  600 - 1319   24 (lens, source) pairs, lens bin outer, x 30 angles
    gg  entries 1320 - 1499   6 lens bins x 30 angles
    gk  entries 1500 - 1679   6 lens bins x 30 angles
    ks  entries 1680 - 1799   4 source bins x 30 angles
    kk  entries 1800 - 1808   9 bandpowers

The entries the mask keeps, per block and per lens bin:

| block | entries | kept entries | kept, lens bin 1 | kept, lens bin 2 | kept, lens bin 3 | kept, lens bin 4 | kept, lens bin 5 | kept, lens bin 6 |
|---|---|---|---|---|---|---|---|---|
| `ss` | 600 | 400 | - | - | - | - | - | - |
| `gs` | 720 | 192 | 40 | 44 | 52 | 56 | 0 | 0 |
| `gg` | 180 | 43 | 9 | 10 | 12 | 12 | 0 | 0 |
| `gk` | 180 | 46 | 10 | 11 | 12 | 13 | 0 | 0 |
| `ks` | 120 | 48 | - | - | - | - | - | - |
| `kk` | 9 | 9 | - | - | - | - | - | - |

The mask keeps entries of every source bin in `ss` and `ks`, and removes every entry of lens bins 5 and 6. A parameter that acts only on lens bins 5-6 therefore does not change the masked data vector of any likelihood of this folder.

# What each combination fixes <a name="desy1xplanck_fixed_params"></a>

| likelihood | `fixed_params` | why |
|---|---|---|
| `cosmic_shear` | none | |
| `combo_2x2pt_ss_sk` | none | |
| `combo_3x2pt_ss_sk_sk` | none | |
| `combo_2x2pt` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6`, `DES_PM5`, `DES_PM6` | the mask removes lens bins 5-6 from `gs` and `gg` |
| `combo_2x2pt_ss_sg` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6`, `DES_PM5`, `DES_PM6` | the mask removes lens bins 5-6 from `gs` |
| `combo_3x2pt` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6`, `DES_PM5`, `DES_PM6` | the mask removes lens bins 5-6 from `gs` and `gg` |
| `combo_3x2pt_ks_gk_kk` | `DES_PM1`, `DES_PM2`, `DES_PM3`, `DES_PM4`, `DES_PM5`, `DES_PM6` | no `gs` block: the point masses act on galaxy-galaxy lensing only |
| `combo_3x2pt_ks_gk_kk` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6` | the mask removes lens bins 5-6 from `gk` |
| `combo_6x2pt` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6`, `DES_PM5`, `DES_PM6` | the mask removes lens bins 5-6 from `gs`, `gg`, and `gk` |

The fixed values are the fiducial values of `params_lens_maglim.yaml`:

| parameter | fixed value | taken from |
|---|---|---|
| `DES_DZ_L5` | 0.002 | center of the prior and of the `ref` |
| `DES_DZ_L6` | 0.002 | center of the prior and of the `ref` |
| `DES_B1_5` | 2.0 | center of the `ref` (the prior is flat, [0.8, 3]) |
| `DES_B1_6` | 2.2 | center of the `ref` (the prior is flat, [0.8, 3]) |
| `DES_DZ2_L5` | 1.08 | center of the prior and of the `ref` |
| `DES_DZ2_L6` | 0.845 | center of the prior and of the `ref` |
| `DES_PM1` ... `DES_PM6` | 0 | center of the prior and of the `ref` |

`DES_BMAG_5` and `DES_BMAG_6` are constants of `params_lens_maglim.yaml` in every likelihood that includes that file. As the bins they act on are removed, the fixed values do not change the data vector; they set only the values the chains record. `DES_B2_5` and `DES_B2_6` are constants (0) of the same file, but their value matters: see the note below the next table.

# Parameters that should not be varied <a name="desy1xplanck_not_varied"></a>

The table lists, for every likelihood of this folder, the parameters that do not change its masked data vector, why, and who fixes them. The columns:

    Applies when   the likelihood option (or parameter value) under which the row holds
    Response       `0`: the masked data vector does not change at all
    Fixed by       `fixed_params` of the likelihood yaml; a parameter file, where the
                   parameter is a constant; user: the parameter files sample it, and the
                   user's yaml must fix it; not declared: the likelihood does not include
                   the parameter's file, so the parameter does not exist there

A `*` in a parameter name stands for every bin number.

| Likelihood | Parameters | Reason | Applies when | Response | Fixed by |
|---|---|---|---|---|---|
| `cosmic_shear` | `DES_DZ_L*`, `DES_DZ2_L*`, `DES_B1_*`, `DES_B2_*`, `DES_BMAG_*`, `DES_PM*` | no lens block | always | `0` | not declared (the yaml includes `params_source.yaml` only) |
| `cosmic_shear` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `cosmic_shear` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `cosmic_shear` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_2x2pt_ss_sk` | `DES_DZ_L*`, `DES_DZ2_L*`, `DES_B1_*`, `DES_B2_*`, `DES_BMAG_*`, `DES_PM*` | no lens block | always | `0` | not declared (the yaml includes `params_source.yaml` only) |
| `combo_2x2pt_ss_sk` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_2x2pt_ss_sk` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_2x2pt_ss_sk` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_3x2pt_ss_sk_sk` | `DES_DZ_L*`, `DES_DZ2_L*`, `DES_B1_*`, `DES_B2_*`, `DES_BMAG_*`, `DES_PM*` | no lens block | always | `0` | not declared (the yaml includes `params_source.yaml` only) |
| `combo_3x2pt_ss_sk_sk` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_3x2pt_ss_sk_sk` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_3x2pt_ss_sk_sk` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_2x2pt` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6`, `DES_PM5`, `DES_PM6` | the mask removes every `gs` and `gg` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `fixed_params` of `combo_2x2pt.yaml` |
| `combo_2x2pt` | `DES_BMAG_5`, `DES_BMAG_6` | the mask removes every `gs` and `gg` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `params_lens_maglim.yaml` |
| `combo_2x2pt` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_2x2pt` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_2x2pt` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_2x2pt_ss_sg` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6`, `DES_PM5`, `DES_PM6` | the mask removes every `gs` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `fixed_params` of `combo_2x2pt_ss_sg.yaml` |
| `combo_2x2pt_ss_sg` | `DES_BMAG_5`, `DES_BMAG_6` | the mask removes every `gs` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `params_lens_maglim.yaml` |
| `combo_2x2pt_ss_sg` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_2x2pt_ss_sg` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_2x2pt_ss_sg` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_3x2pt` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6`, `DES_PM5`, `DES_PM6` | the mask removes every `gs` and `gg` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `fixed_params` of `combo_3x2pt.yaml` |
| `combo_3x2pt` | `DES_BMAG_5`, `DES_BMAG_6` | the mask removes every `gs` and `gg` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `params_lens_maglim.yaml` |
| `combo_3x2pt` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_3x2pt` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_3x2pt` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_3x2pt_ks_gk_kk` | `DES_PM1`, `DES_PM2`, `DES_PM3`, `DES_PM4`, `DES_PM5`, `DES_PM6` | no `gs` block: the point masses act on galaxy-galaxy lensing only | always | `0` | `fixed_params` of `combo_3x2pt_ks_gk_kk.yaml` |
| `combo_3x2pt_ks_gk_kk` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6` | the mask removes every `gk` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `fixed_params` of `combo_3x2pt_ks_gk_kk.yaml` |
| `combo_3x2pt_ks_gk_kk` | `DES_BMAG_5`, `DES_BMAG_6` | the mask removes every `gk` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `params_lens_maglim.yaml` |
| `combo_3x2pt_ks_gk_kk` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_3x2pt_ks_gk_kk` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_3x2pt_ks_gk_kk` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_6x2pt` | `DES_DZ_L5`, `DES_DZ_L6`, `DES_B1_5`, `DES_B1_6`, `DES_DZ2_L5`, `DES_DZ2_L6`, `DES_PM5`, `DES_PM6` | the mask removes every `gs`, `gg` and `gk` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `fixed_params` of `combo_6x2pt.yaml` |
| `combo_6x2pt` | `DES_BMAG_5`, `DES_BMAG_6` | the mask removes every `gs`, `gg` and `gk` entry of lens bins 5-6 | `data_file: Y3xPlanckPR4.dataset` | `0` | `params_lens_maglim.yaml` |
| `combo_6x2pt` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_6x2pt` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_6x2pt` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |

`DES_B2_*` are constants (0) of `params_lens_maglim.yaml` but are not in the table: a nonzero value in any lens bin switches on the one-loop bias terms of every lens bin, so even `DES_B2_5` and `DES_B2_6` change the data vector although the mask removes their bins.

The likelihood also turns `use_baryon_pca` off when `external_baryon_suppression: True` (the `bfmt` theory block) or `create_baryon_pca: True` is set, so the `DES_BARYON_Q*` rows hold in those runs too.

The examples declare `DES_BARYON_Q1` and `DES_BARYON_Q2` in their own `params` block, which overrides the constants of `params_source.yaml`. The examples with `use_baryon_pca: false` (`EXAMPLE_EVALUATE1.yaml`, `EXAMPLE_EVALUATE2.yaml`, `EXAMPLE_MCMC2.yaml`, and the matching EMUL2 examples) declare them as constants (`value: 0.0`); the examples with `use_baryon_pca: True` (`EXAMPLE_MCMC1.yaml`, `EXAMPLE_EMUL2_MCMC1.yaml`, `EXAMPLE_EMUL2_POLY1.yaml`) sample them with priors. The examples `EXAMPLE_EVALUATE1.yaml`, `EXAMPLE_EVALUATE2.yaml`, `EXAMPLE_EMUL2_EVALUATE1.yaml`, and `EXAMPLE_EMUL2_EVALUATE2.yaml` run `IA_model: 0` and sample `DES_A2_1`, `DES_A2_2`, and `DES_BTA_1` as `params_source.yaml` does; `EXAMPLE_EMUL2_MCMC2.yaml`, also NLA, fixes them in its own `params` block.

# Changing the mask, the scale cuts, or the probes <a name="desy1xplanck_changing"></a>

> [!Warning]
> The `fixed_params` blocks encode the mask of `Y3xPlanckPR4.dataset` and the blocks of each combination. A user who changes the mask or the scale cuts (for example a mask that keeps lens bins 5-6), or the probes of a combination, must revisit that combination's `fixed_params`: a parameter it fixes may then act on the data vector, and the fixed value hides that dependence without any error.

A `fixed_params` block in the likelihood block of the user's yaml replaces the combination's block as a whole: `fixed_params: null` samples every parameter again, and a shorter block keeps only the entries it repeats.

The steps below assume the Conda cocoa environment is active (`conda activate cocoa`), the shell is bash, and the current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: in the likelihood block of the user's yaml (below, a copy of `EXAMPLE_EVALUATE2.yaml` named `MY_EVALUATE2.yaml`), set `fixed_params`

    likelihood:
      desy1xplanck.combo_6x2pt:
        path: ./external_modules/data/desy1xplanck
        data_file: Y3xPlanckPR4.dataset
        fixed_params: null

or, to keep part of the combination's block (here the point masses of `combo_3x2pt_ks_gk_kk` while lens bins 5-6 are sampled),

    likelihood:
      desy1xplanck.combo_3x2pt_ks_gk_kk:
        path: ./external_modules/data/desy1xplanck
        data_file: Y3xPlanckPR4.dataset
        fixed_params:
          DES_PM1:
            value: 0.0
          (...)
          DES_PM6:
            value: 0.0

**Step :three:**: add a value for every parameter sampled again to the `sampler: evaluate: override` block (cobaya's evaluate sampler refuses an override of a parameter that is not sampled, so the block must match the sampled parameters)

    sampler:
      evaluate:
        override:
          (...)
          DES_DZ_L5: 0.002
          DES_B1_5: 2.0

**Step :four:**: run the evaluation

    cobaya-run ./projects/desy1xplanck/MY_EVALUATE2.yaml -f

> [!NOTE]
> A parameter declared with a `prior` in the `params` block of the user's yaml is sampled whatever `fixed_params` holds: the user's `params` entry takes precedence over the likelihood defaults. `EXAMPLE_MCMC2.yaml` and `EXAMPLE_EMUL2_MCMC2.yaml` declare the nuisance parameters in their own `params` block, and leave out the ones `combo_6x2pt` fixes.
