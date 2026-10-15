# DES × Planck: optional n(z) reconstruction study

This study asks whether the interpretation of a supplied histogram changes
predictions. It is **run explicitly**, not collected by routine pytest and
not compared with a frozen expected output vector. The small compiled-reader
checks remain part of software regression testing.

## Four reconstructions of the same histogram

`photoz_histogram_akima: 0` is the unchanged default: a curve through the
original bin centres, followed by a natural cubic on a uniform lookup grid.
With `photoz_histogram_akima: 1`, choose `photoz_histogram_akima_factor: 3`,
`5` or `7`. These odd factors retain the centre of each uniform input bin.
Irregular input cells are remapped with overlap-weighted cell averages.
Akima then smooths the transitions between those prepared cells.

The **preparation factor is fixed independently of accuracy**. Raising the
lookup accuracy subdivides the same Akima polynomial algebraically; it does
not refit or sharpen the curve. Both reconstructions always use uniform-grid
lookup; the former compile-time GSL bypass has been retired.

![Original, 3-, 5- and 7-point preparation on an illustrative histogram](../nz_akima_preparation.png)

This first figure is a **toy histogram**, with coordinates measured in original
bin widths rather than redshift. Its lower row always shows the prescribed
edge at coordinate 3, from 2.45 to 3.55, including both neighbouring cell
centres. Each row uses the same axes across all four columns. This fixed
illustration is separate from the automatic survey close-up explained below.

The next figure has only two panels: an actual survey histogram edge, and
redistribution of probability among original cells for every source/lens bin.
It samples the compiled C readers at zero photo-z shift, unit stretch and
lookup boost 5; Python does not fit a replacement curve. The shared plotter
is `cosmolike_notebook_utils/plot_nz_reconstruction.py` in CosmoLike core.
The study regenerates this figure automatically after measuring the curves.

![Four compiled reconstructions and their bin-area changes](results/reconstructions.png)

**How the panels are constructed.** The left close-up follows the same
rule for every project:

1. Among all source/lens distributions, select the one with the largest
   probability redistribution under the **original cubic**. This selection
   does not depend on which Akima curve differs most.
2. Divide each input cell's normalized probability by its width to obtain
   histogram heights, then find the largest absolute jump between adjacent
   heights. This chooses an interior **bin edge**, not a bin centre.
3. Show two input cells before that edge and three after it, clipping the
   window to the supplied histogram bounds. Exact ties use the first
   distribution or edge in the saved order.
4. Overlay the saved compiled-reader samples for all four reconstructions
   on that same window. No new spline is fitted for display. The vertical
   range includes the histogram and all four curves, including negative
   ringing or boundary cutoffs, with a small margin.

The right panel includes **every** source/lens distribution, independently
of the close-up selection. Its bars show movement of normalized probability
between original cells; total-area changes are reported separately below.

## Model-vector distances

All four arms use the same stored NLA example2 point, histogram files,
mask and supplied covariance. They use `accuracyboost: 5` and
`integration_accuracy: 7`: 8,192 nodes per Limber panel. All quadrature
rules are capped at 8,192, including the source/lens mean-redshift rules.
These are deliberately expensive study settings, not new likelihood defaults.

For two model vectors, Δχ² = (d_b − d_a)ᵀ C⁻¹ (d_b − d_a).
This is a distance between predictions, not a difference in their fit to data.
Pairwise squared distances are not additive.

| Original → 3 | 3 → 5 | 3 → 7 | 5 → 7 |
|---:|---:|---:|---:|
| 0.1304652 | 0.002228624 | 0.002972851 | 0.0001359452 |

The study adopts practical numerical stability relative to Δχ² = 0.2;
it does not require an exact continuum limit. The earlier boost scan is
preserved in the core study record. Smaller 5→7 distances indicate stability
of this sequence at the tested point, not proof that Akima is physically
more accurate or that every cosmology has converged. Covariance construction
with an Akima kernel requires its own numerical checks.

## Original at defaults versus original refined

This comparison changes only `accuracyboost` and `integration_accuracy`.
The default values are read from the current likelihood YAML; all physical
settings and inputs remain the same stored example2 case.

| Original setting | `accuracyboost` | `integration_accuracy` |
|---|---:|---:|
| YAML defaults | 1 | 0 |
| Refined study | 5 | 7 |

**Original default → refined: Δχ² = 0.022126581.**
This measures numerical accuracy without changing the histogram interpretation.

## Area accounting

At zero photo-z shift and unit stretch, integrate each compiled curve over
every original histogram cell. Split at all polynomial knots: independent
two- and four-point Gauss rules agree to 2×10⁻¹³ absolute per cell.
The table ranges include every source/lens tomographic distribution.

| Reconstruction | Total-area range | Largest redistributed fraction |
|---|---:|---:|
| Original cubic | 0.998939123–0.999981221 | 0.256785% |
| 3-point Akima | 1–1 | 0.105388% |
| 5-point Akima | 1–1 | 0.063233% |
| 7-point Akima | 1–1 | 0.0451665% |

The redistributed fraction is 50 Σ|q/Σq − p| percent, where p is the
normalized input cell mass and q is the reconstructed cell area. Normalizing
q in this diagnostic separates movement between bins from total-area drift;
**it does not renormalize the production reader**. Akima preserves total area
for these uniform-bin constructions but still exchanges mass between cells.
That statement does not extend to arbitrary irregular input. Original signed
input cells are retained, and neither arm adds clipping or area corrections.

For DES cluster, this switch acts on the galaxy source/lens histograms.
Its separately supplied cluster selection probabilities retain their linear
interpolation; they are not galaxy histogram densities.

## Run only when investigating reconstruction

From an activated Cocoa environment and the Cocoa directory, run the
**opt-in pytest test** explicitly:

```bash
python -m pytest projects/desy1xplanck/tests/studies/study_nz_reconstruction.py -v -s
```

The `study_` filename deliberately stays outside normal pytest discovery.
This test launches a fresh Python process, computes the four reconstructions
and the original-default comparison, checks the shared inputs/covariance and
cache round trip, integrates the cell areas and regenerates the figure. It
uses boost 5 and integration level 7, without a frozen expected vector or
an asserted model-choice Δχ². A numerical/check failure fails the test.
Results are written in the temporary directory printed at completion.

For an explicitly chosen persistent output directory, run the same study
script directly:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python projects/desy1xplanck/tests/studies/run_nz_reconstruction.py \
  --output /tmp/desy1xplanck-nz-study
```

Use a new output directory: existing vectors are never overwritten. The
script checks finite values, positive-definite selected inverse covariance,
unchanged inputs and covariance, and the original→3→5→7→original cache round
trip. It reports measured distances without asserting a frozen Δχ².
`--part defaults`, `--part refined` and `--part compare` allow separate stages.
`--boost` and `--integration-accuracy` allow deliberate numerical follow-ups;
the first-stage factors remain 3, 5 and 7.

To redraw the committed figure without a cosmology calculation:

```bash
python projects/desy1xplanck/tests/generate_nz_reconstruction_figures.py
```

[Comparison record](results/comparison.json), [default settings](results/defaults.json),
[refined settings and input fingerprints](results/refined.json),
[area audit](results/probability.json), and their adjacent NPZ arrays preserve
the measurements. The inverse covariance is stored once through Git LFS.
DES Y6 has no statistical row because its supplied identity metric is not a
physical survey covariance.
