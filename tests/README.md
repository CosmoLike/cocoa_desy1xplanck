# Unit tests for the desy1xplanck likelihoods

These tests catch two kinds of silent breakage: a chi2 that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads).

Every model build runs in its own worker subprocess. In this project
both examples share one data set, so the isolation is preventive: it
keeps the suite immune to the process abort that different data-set
dimensions trigger inside cosmolike (desy1xplanck has that layout), and
every project keeps one architecture. The commands below stay the
same.

## Running the tests

From the `Cocoa/` folder, with the cocoa conda environment active and
`start_cocoa.sh` sourced:

    python -m pytest ./projects/desy1xplanck/tests

Without pytest:

    python -m unittest discover -s ./projects/desy1xplanck/tests -v

The suite changes no project files. Each test streams a progress line
per model build and per evaluation, then a report block with the
computed chi2, the stored reference, the difference, and the pass
limit. A full run performs about 50 likelihood evaluations and takes a
few minutes. The test modules force `OMP_NUM_THREADS=4` internally.
The suite never waits for a keypress: a space/enter prompt between
tests means the output is being piped through a pager such as `less`,
so run the command with nothing piped after it.

## The eight tests

1. `test_1`: chi2 of the cosmic-shear likelihood at a fixed reference
   point must stay within 0.2 of the value stored in
   `frozen/reference_chi2.json`.
2. `test_2`: on one model, that point is evaluated fresh and then
   again as the 10th of 10 cosmologies in a row; the two chi2 values
   must agree to 1e-4. Leftover state or an OpenMP race breaks the
   agreement.
3. `test_3`: same as test 1 with the TATT intrinsic-alignment model
   (`IA_model: 1`) and `DES_A2_1=0.05`, `DES_BTA_1=0.05`,
   `DES_A2_2=-1.51541`.
4. `test_4`: same as test 2 with the TATT model.
5. -8. the same four tests for the 6x2pt likelihood.
9. -14. `test_example2_2x2pt.py` (numbered 11-14): the four standard
   tests on `desy1xplanck.combo_2x2pt` (example2 with the probe selection
   reduced to galaxy clustering plus galaxy-galaxy lensing).

Accuracy checks (`test_accuracy.py`): first a one-knob-at-a-time scan
on the 6x2pt NLA configuration, then six all-knobs checks (A1-A6):
the three probes with both IA models re-evaluated with the numerical
settings pushed far beyond the defaults (cosmolike accuracyboost 2,
integration_accuracy 10, lmax 200000, kmax_boltzmann 40; CAMB
AccuracyBoost 2, k_per_logint 50, kmax 50). The scan keeps an
accuracyboost 5 entry as a deliberate stress knob: in this project it
breaks the 6x2pt integration tables and shifts the chi2 by +27, so it
stays out of the all-knobs set. Each check reports
delta chi2 = chi2(high accuracy) - chi2(default, frozen), no
pass/fail. High-accuracy evaluations take minutes; skip the file with
`--ignore ./projects/desy1xplanck/tests/test_accuracy.py`.

This project's shipped data vector is REAL data, and the example
cosmology is not its best fit, so the chi2 there sits far from the
minimum, where it responds linearly to tiny numerical changes. Every
variant therefore evaluates against a data vector generated at the
fiducial point during the freeze: `frozen/data/synthetic_desy1xplanck.dataset`
(default NLA model) for the NLA tests and
`frozen/data/tatt_desy1xplanck.dataset` (TATT model) for the TATT tests.
Both come from the example2 (6x2pt) model, whose full-length vector
serves every probe; at its own minimum the chi2 response is quadratic
and the drift and accuracy numbers stay meaningful.

## Why the tests keep their own copy of everything

The tests read nothing from the live project: not `../data`, not the
`EXAMPLE_EVALUATE` yaml files, and not the likelihood default yaml
files. Instead, `frozen/` holds:

- `frozen_config_example{1,2}.py`: the complete cobaya configuration
  as a yaml string plus the exact evaluation point. Every option and
  every parameter is written out, including the ones that normally
  come from `params_source.yaml` and the other default files, so
  editing those files cannot change what the tests evaluate.
- `data/`: the tests' own copy of the data vectors, covariance, n(z),
  and masks.
- `EXAMPLE_EVALUATE{1,2}.yaml`: snapshots kept only so a human can
  diff how the live examples drifted since the freeze.

`manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every frozen file. Each test
verifies the manifest first and refuses to run when a frozen file was
edited, naming the file. The result: users may change the live data
and examples freely, and nobody can quietly edit the frozen state
either.

## Refreshing the frozen state (maintainers only)

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze:

    python ./projects/desy1xplanck/tests/generate_frozen_reference.py --overwrite

Run it from the `Cocoa/` folder with the environment set up as above.
It rebuilds `frozen/` from the current project, prints the four new
reference chi2 values, and rewrites the manifest. Review the printed
chi2 values against the old references before committing: they define
what every later test run compares against.
