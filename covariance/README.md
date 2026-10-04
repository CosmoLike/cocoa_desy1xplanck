# Table of contents

1. [Overview](#overview)
2. [Running the covariance notebook](#running)
3. [Changing the covariance accuracy](#accuracy)
4. [Reading the figures](#figures)
5. [Running the tests](#tests)
6. [Files](#files)
7. [Appendix](#appendix)
   1. [FAQ: Which survey does the example use?](#survey)
   2. [FAQ: What does the calculation include?](#gaussian)
   3. [FAQ: How can users check convergence?](#convergence)
   4. [FAQ: How can users reuse the calculation?](#reuse)

# Overview <a name="overview"></a>

[EXAMPLE_EVALUATE_COVARIANCE.ipynb](../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
computes an analogous real-space 3×2pt covariance for
[the supplied dataset](../data/Y3xPlanckPR4.dataset). It keeps Gaussian (G),
super-sample (SSC), connected non-Gaussian (cNG) and total matrices separately.
The example has 1,500 entries before cuts and 635 after the dataset mask.
The shared reader applies that mask to both axes of every component and to
the supplied total; the notebook plots their correlation matrices together.

The supplied file has 1,809 entries, but this generator currently computes only
its first 1,500 galaxy/shear entries. Their 635 retained entries form the
supported 3×2pt submatrix. CMB lensing auto- and cross-covariances are not
generated. The supplied full joint matrix remains available to the likelihood.

The default computes the native measurement at boost 1. Users can request a
second boost for numerical comparisons and a companion measurement space.
The supplied matrix is read only for comparison; no likelihood files are changed.

> [!NOTE]
> The forecast uses massless neutrinos, Limber spectra, linear galaxy bias,
> zero IA, magnification and RSD, and a spherical-cap footprint. SSC uses the
> isotropic halo response and cNG the halo trispectrum. All-pairs non-Limber
> covariance and massive-neutrino non-Gaussian terms are not implemented.
> These physical choices differ from the supplied likelihood matrices.
> Matching their measurement layout does not establish physical or numerical
> equivalence.

# Running the covariance notebook <a name="running"></a>

We assume Cocoa and the DES × Planck galaxy–shear block project are installed, users have run
`conda activate cocoa`, the shell is Bash, and the current folder is
`cocoa/Cocoa`. The notebook uses the Python environment activated by Cocoa.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile the DES × Planck galaxy–shear block interface, including the covariance components.

    unset IGNORE_COSMOLIKE_DESY1XPLANCK_CODE
    source ./projects/desy1xplanck/scripts/compile_desy1xplanck.sh

**Step :three:**: start Jupyter.

    jupyter notebook --no-browser --port=8888

**Step :four:**: open the URL printed by Jupyter and select
`projects/desy1xplanck/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

**Step :five:**: select **Kernel → Restart Kernel and Run All Cells**.

The notebook starts with `boosts = [1]` and `spaces = ["real"]`.
It computes the native matrix, applies the selected dataset's mask, reports
positivity and plots the computed components and supplied total. Set
`boosts = [1, 2]` to add the numerical-refinement comparison.

| Output in `covariance/` | Contents |
| --- | --- |
| `forecast_real.npz` | Full computed G, SSC, cNG, total, ordering and settings. |
| `forecast_likelihood_selection.npz` | Cut components, supplied total, original entry indices and probe labels. |
| `forecast_camb.npz` | CAMB tables used by the native calculation. |

Rerunning the final cell replaces these computed output files.

> [!NOTE]
> The notebook assigns eight threads to CosmoLike's OpenMP loops and
> one thread to BLAS. Change `ci.set_omp_threads(n=8)` in the notebook
> if fewer cores are available. Run one calculation at a time when
> measuring execution time.

> [!TIP]
> To inspect the forecast inputs before running CAMB, see
> [which survey the example uses](#survey).

# Changing the covariance accuracy <a name="accuracy"></a>

We assume users have run `conda activate cocoa`, use Bash, and are in
`cocoa/Cocoa`. The interface must already be compiled.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: start Jupyter and open
`projects/desy1xplanck/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

    jupyter notebook --no-browser --port=8888

**Step :three:**: set the first calculation's accuracy in its configuration cell.

```python
boosts = [1, 2]
settings = survey.configuration(accuracy_boost=boosts[0])
```

The numerical baseline is in [`default.yaml`](default.yaml). The notebook
loads that file through `survey.configuration`. Its internal refinements
multiply the global boost: factors 2 and 3 at boost 1 become 4 and 6 at
boost 2. Resolved settings save both the base controls and effective grids.
`integration_accuracy` is independent of the global boost. Its levels
0/1/2/3/4 select precomputed GSL rules with 96/128/256/512/1024 nodes per
panel for radial, mass and angular integrals. Wide angular bins are split
into panels to resolve high-multipole oscillations. No rule below 64 nodes
is supported, including in low-level calls. To check quadrature alone, use
`survey.configuration(accuracy_boost=1, integration_accuracy=1)` and compare
with level zero. A convergence scan should include levels 2, 3 and 4:
compare the default directly with level 4, then check that the final 3-to-4
refinement is small. Keep interpolation settings fixed in this scan and
check them separately. The defaults are being checked against refined full
matrices; a higher level alone is not a convergence certificate.

`accuracy_boost` controls interpolation and cutoffs. Supported values are
1, 2, 4 and 8. It raises the covariance multipole cutoffs and refines the non-Gaussian,
lensing-window and shared core interpolation tables. It leaves
CAMB and data-vector accuracy settings unchanged.

**Step :four:**: choose which values to compare in the refinement cell.

```python
boosts = [1, 2, 4]
```

For the more expensive comparison, include boost 8:

```python
boosts = [1, 2, 4, 8]
```

**Step :five:**: restart the kernel and run all cells.

The calculation keeps the cosmology, galaxy distributions, noise, angular
bins and Fourier-band endpoints fixed. Only the covariance accuracy changes. The highest
boost in the comparison supplies the reference matrix for the difference
plots and variance-ratio table.

> [!NOTE]
> A larger boost is a numerical resolution, not a guaranteed survey
> accuracy. Boost 1 uses the project baseline. Boost 8 uses more modes and
> table samples and can require substantially more memory and time.
> See [how to check convergence](#convergence).

# Reading the figures <a name="figures"></a>

| Figure | What it teaches |
| --- | --- |
| Split-triangle correlation matrix | Compare the generated native-space covariance in the lower triangle with the supplied likelihood covariance in the upper triangle, after the same cuts. Each uses its own diagonal normalization. |
| G, SSC and cNG maps and histograms | Compare each component after normalization by the total diagonal variances. |
| Error changes | With multiple boosts, compare first-source-bin standard deviations with the highest tested boost, in percent, for the native measurement. |
| Generalized-mode report | Bound variance changes over every linear combination of measurements. |

The correlation comparison follows the layout of
[Friedrich et al. (2021), Fig. 6](https://arxiv.org/abs/2012.08568).
The component maps and histogram adapt
[Barreira, Krause & Schmidt (2018), Fig. 1](https://arxiv.org/abs/1807.04266).
These figures display the notebook's calculation, not data from the papers.

Negative correlations remain visible. A grey cell in an element-ratio
map means its denominator is zero or too small for the selected cutoff.
The component titles count undefined plotting ratios, not entries removed
by the likelihood mask. No plot clips eigenvalues or adjusts the covariance.

# Running the tests <a name="tests"></a>

We assume users have run `conda activate cocoa`, use Bash, and are in
`cocoa/Cocoa`, with the DES × Planck galaxy–shear block interface compiled.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run the covariance tests.

    python -m pytest projects/desy1xplanck/tests/covariance

The tests check this project's input layout, real/Fourier assembly,
thread repeatability, subset positivity and saved metadata. The [covariance test guide](../tests/covariance/README.md)
describes each check.

For the ordinary data-vector tests, we again assume the activated Conda
Cocoa environment, Bash, and the current folder `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run only the data-vector tests.

    python -m pytest projects/desy1xplanck/tests/data_vector

These tests compare likelihood predictions with the stored references.
The [data-vector test guide](../tests/data_vector/README.md) explains them.

# Files <a name="files"></a>

| File or folder | Purpose |
| --- | --- |
| [EXAMPLE_EVALUATE_COVARIANCE.ipynb](../EXAMPLE_EVALUATE_COVARIANCE.ipynb) | Run, refine and plot real/Fourier G, SSC and cNG matrices. |
| [desy1xplanck_covariance.py](desy1xplanck_covariance.py) | Specify survey inputs, initialize this project's interface and call the shared calculation. |
| [Shared covariance package](../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/covariance/README.md) | Reuse integration preparation, Gaussian assembly, halo inputs, accuracy settings and diagnostics. |
| [Shared plotting script](../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/plot_covariances.py) | Plot covariance arrays from any project. |
| [Covariance C files](../../../external_modules/code/cosmolike_core/cosmolike/covariances/README.md) | Read the physics and each compiled component's role. |

# Appendix <a name="appendix"></a>

## FAQ: Which survey does the example use? <a name="survey"></a>

The redshift files are `nz_maglim_Y3_unblinded_02_26_21.txt` and `nz_source_Y3_unblinded_02_26_21.txt` in `data/`.
Each column supplies a bin's radial shape. The catalog densities are
specified separately in [desy1xplanck_covariance.py](desy1xplanck_covariance.py).

The source densities and per-component shape dispersions are taken from
the NGAL and SIG_E headers of the DES Y3 two-point release
`2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits`.
The area is represented by a spherical cap of 4,143 square degrees.
The [DES Y3 covariance analysis](https://arxiv.org/abs/2012.08568)
explains why catalog weights and footprint geometry matter; the cap is
an approximation to that footprint.
The MagLim lens densities follow
[Porredon et al., Table 1](https://academic.oup.com/mnras/article/511/2/2665/6516441).
The redshift files and angular bins follow the project's Y3×Planck dataset.
CMB lensing, reconstruction noise and its cross blocks are not included;
this example computes the galaxy–shear block only.

This uses the current Y3×Planck dataset: six MagLim lens and four source
bins. The angular bins follow that dataset. The Fourier bands are a
companion example. CMB lensing covariance is not implemented here.

| Survey input | Example choice |
| --- | --- |
| Area | 4143 square degrees |
| Lens bins | 6 |
| Source bins | 4 |
| Angular bins | 30 logarithmic bins, 0.25–250 arcminutes |
| Integer Fourier bands | 15 bands, multipoles 30–4000 |
| Real-space vector | 1,500 entries |
| Fourier vector | 600 entries |
| Neutrino mass | Zero |
| Intrinsic alignment | Zero |
| Photo-z shifts | Zero |
| Magnification | Zero |

The adapter lists every density, shape dispersion, galaxy bias and
cosmological input. The notebook prints those settings and archives them
with each matrix. The examples do not replace a supplied likelihood matrix.

## FAQ: What does the calculation include? <a name="gaussian"></a>

For Gaussian fields, a four-point expectation separates into products
of two-point expectations. A covariance between measured spectra AB and
CD therefore needs AC, BD, AD and BC spectra, even if those crossed pairs
are excluded from the measured data vector. The example retains the
complete field matrix before assembling the measured rows.

The signal uses nonlinear matter power and Limber projection. Spherical
spin operators average the resulting angular spectra over each angular
bin. The signal covariance uses the $`f_{\rm sky}`$ approximation.

Pure white noise extends to arbitrarily high multipoles. The calculation
replaces that infinite sum with the analytic pair-noise expression using
a spherical-cap footprint. Signal and mixed signal-noise terms retain
the finite multipole sum. A cap correction to noise pair counts does not
make the signal covariance an exact treatment of an irregular footprint.

Fourier bandpowers include pure noise inside the finite band sum. They
average the core spectra directly. Real-space means retain Cocoa's
extra source-leg factor for matching its angular-transform convention.
Each convention is applied consistently to G, SSC and cNG signal terms;
white noise receives neither source conversion.

SSC describes correlations induced by modes larger than the survey;
cNG describes the connected four-point contribution inside it.
The notebook computes both terms for every measured cross block. SSC
uses common shell responses before their weighted outer products; cNG
uses the 1-halo, 2-halo (1+3), 2-halo (2+2), 3-halo and 4-halo terms.
These are approximations, not a simulation calibration. See
[Krause & Eifler](https://arxiv.org/abs/1601.05779) and
[Takada & Hu](https://arxiv.org/abs/1302.6994) for the physical decomposition.

## FAQ: How can users check convergence? <a name="convergence"></a>

Compare successive boosts with the same physical inputs. The notebook
prints the largest departure of the generalized covariance eigenvalues
from one. They solve

```math
C_b v = \lambda C_{\rm ref} v.
```

The smallest and largest eigenvalues bound the variance ratio for any
linear combination of measurements. This checks coupled directions that
a diagonal comparison alone can miss. Raw largest covariance eigenvalues
do not identify the most informative cosmological modes.

Every total matrix must have nonnegative variance in every direction;
an invertible total must be positive definite. A positive diagonal alone
is insufficient. The notebook reports eigenvalues without repairing them.

The reference boost is only the highest resolution tested. Establish its
stability with further refinement, then assess marginalized parameter
errors and the Fisher Figure of Merit for the intended survey. The
likelihood's data-vector $`\lvert\Delta\chi^2\rvert<0.2`$ rule is not a
covariance convergence requirement.

## FAQ: How can users reuse the calculation? <a name="reuse"></a>

A project supplies its own initialized interface, redshift files, survey
numbers and physical choices. The shared Python package prepares arrays
and asks the covariance C components to perform the integration and
projection. Copy the small survey adapter, not those algorithms.

Other project interfaces must link the same covariance sources before
running these examples. Cluster observables need their own physical
model; changing the number of galaxy bins does not create a cluster
covariance.

Rectangular projection inputs allow Python to request matrix subblocks.
The C routines use OpenMP inside one process and never start MPI work.
A future Python dispatcher can distribute those subblocks while keeping
all cross correlations in the assembled matrix.
