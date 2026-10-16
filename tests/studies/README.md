# DES × Planck: optional n(z) reconstruction study

## :interrobang: FAQ: What are optional studies? <a name="optional_studies"></a>

Optional studies explore how numerical settings and modeling choices
affect predictions; they can be expensive, so they run separately from
the routine tests. The shared guide to this n(z) reconstruction study -
the reconstruction definitions, the distance metric, the area
accounting, the figure construction and the run options - is
[The n(z) reconstruction study](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/cosmolike_notebook_utils/README.md#nz_study)
in the CosmoLike core. This file records this survey's measurements.

## The survey figure

![Four compiled reconstructions and their bin-area changes](results/reconstructions.png)

## Measured model-vector distances

All four arms use the stored NLA example point, this survey's histogram
files, mask and supplied covariance, at the shared study settings
(`accuracyboost: 5`, `integration_accuracy: 7`); the distance is the
guide's $`\Delta\chi^2`$, judged against practical stability at 0.2.

| Original → 3 | 3 → 5 | 3 → 7 | 5 → 7 |
|---:|---:|---:|---:|
| 0.1304652 | 0.002228624 | 0.002972851 | 0.0001359452 |

Original at its YAML defaults (`accuracyboost: 1`,
`integration_accuracy: 0`) versus the refined study settings:
**Δχ² = 0.022126581** - numerical accuracy alone, without changing the
histogram interpretation.

## Measured area accounting

| Reconstruction | Total-area range | Largest redistributed fraction |
|---|---:|---:|
| Original cubic | 0.998939123–0.999981221 | 0.256785% |
| 3-point Akima | 1–1 | 0.105388% |
| 5-point Akima | 1–1 | 0.063233% |
| 7-point Akima | 1–1 | 0.0451665% |

## Run the study <a name="run_study"></a>

The environment setup, stage selection and accuracy options are in the
shared guide ([Run the study](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/cosmolike_notebook_utils/README.md#nz_run)). For this survey, from
`cocoa/Cocoa/` with the cocoa environment active:

```bash
python -m pytest projects/desy1xplanck/tests/studies/study_nz_reconstruction.py -v -s
```

or, keeping the results in a chosen folder:

```bash
python projects/desy1xplanck/tests/studies/run_nz_reconstruction.py --output <folder>
```

To redraw the committed figure without a cosmology calculation:

```bash
python projects/desy1xplanck/tests/studies/generate_nz_reconstruction_figures.py
```

[Comparison record](results/comparison.json), [default settings](results/defaults.json),
[refined settings and input fingerprints](results/refined.json),
[area audit](results/probability.json), and their adjacent NPZ arrays preserve
the measurements. The inverse covariance is stored once through Git LFS.

