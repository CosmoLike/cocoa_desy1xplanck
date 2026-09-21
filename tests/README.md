# Unit tests for the desy1xplanck likelihoods

These tests catch two kinds of silent breakage: a $\chi^2$ that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads).

Every model build runs in its own worker subprocess. In this project
both examples share one data set, so the isolation is preventive: it
keeps the tests immune to the process abort that different data-set
dimensions trigger inside cosmolike (desy1xplanck has that layout), and
every project keeps one architecture. The commands below stay the
same.

## Running the tests

From the `Cocoa/` folder, with the cocoa conda environment active and
`start_cocoa.sh` sourced:

    python -m pytest ./projects/desy1xplanck/tests

Without pytest:

    python -m unittest discover -s ./projects/desy1xplanck/tests -v

The tests change no project files. Each test prints a progress line
per model build and per evaluation, then a report with the
computed $\chi^2$, the stored reference, the difference, and the pass
limit.

A full run performs about 50 likelihood evaluations and takes a
few minutes. The test files force `OMP_NUM_THREADS=4` internally.

> [!NOTE]
> The tests never stop to ask for input. If the terminal pauses
> until space or enter is pressed, something sent the output through
> `less` (a program that stops after each full screen): run the
> commands exactly as written above, with nothing added after them.

## The tests

The standard configurations get four tests each: a $\chi^2$ drift check
and a race check, both in the NLA and in the TATT intrinsic-alignment
model. The TATT variants set

    IA_model: 1
    DES_A2_1: 0.05
    DES_BTA_1: 0.05
    DES_A2_2: -1.51541

| check | pass limit                                        | a failure means                    |
|-------|---------------------------------------------------|------------------------------------|
| $\chi^2$  | within 0.2 of `frozen/reference_chi2.json`        | code or data changed the numbers   |
| race  | fresh vs 10th of 10 cosmologies in a row, to $10^{-4}$ | leftover state or an OpenMP race   |

| tests | file | configuration |
|-------|------|---------------|
| 1-4   | `test_example1.py` | cosmic shear (example1) |
| 5-8   | `test_example2.py` | 6x2pt (example2) |
| 11-14 | `test_example2_2x2pt.py` | 2x2pt (`desy1xplanck.combo_2x2pt`: example2 reduced to galaxy clustering plus galaxy-galaxy lensing) |

### Accuracy checks (`test_accuracy.py`, A1-A6)

First a one-knob-at-a-time scan on the 6x2pt NLA configuration, then six all-knobs checks (A1-A6):
the three probes with both IA models re-evaluated with every setting
pushed far beyond the defaults at once. The scan keeps an
`accuracyboost: 5` entry as a deliberate stress knob: in this project
it breaks the 6x2pt integration tables and shifts the $\chi^2$ by
+27, so it stays out of the all-knobs set below.

    # cosmolike likelihood settings
    accuracyboost: 2
    integration_accuracy: 10
    lmax: 200000
    kmax_boltzmann: 40
    # CAMB extra_args (kmax moves with kmax_boltzmann: one physical cutoff)
    AccuracyBoost: 2
    k_per_logint: 50
    kmax: 50

Each check reports $\Delta\chi^2 = \chi^2(\text{high accuracy}) -
\chi^2(\text{default})$: the numerical error of the default
settings. No pass/fail. High-accuracy evaluations take minutes; run
the file on its own, or skip it with

    python -m pytest ./projects/desy1xplanck/tests --ignore ./projects/desy1xplanck/tests/test_accuracy.py

### Synthetic data vectors

This project's shipped data vector is REAL data, and the example
cosmology is not its best fit, so the $\chi^2$ there sits far from the
minimum, where it responds linearly to tiny numerical changes. Every
variant therefore evaluates against a data vector generated at the
fiducial point during the freeze: `frozen/data/synthetic_desy1xplanck.dataset`
(default NLA model) for the NLA tests and
`frozen/data/tatt_desy1xplanck.dataset` (TATT model) for the TATT tests.
Both come from the example2 (6x2pt) model, whose full-length vector
serves every probe; at its own minimum the $\chi^2$ response is quadratic
and the drift and accuracy numbers stay meaningful.

## Tests keep their own copy of configurations and data

The tests read nothing from the live project: not `../data`, not the
`EXAMPLE_EVALUATE` yaml files, and not the likelihood default yaml
files. Instead, `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_example{1,2}.py` | the complete cobaya configuration as a yaml string, plus the exact evaluation point |
| `data/` | the tests' own copy of the data vectors, covariance, n(z), and masks |
| `EXAMPLE_EVALUATE{1,2}.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

In the configuration modules every option and every parameter is
written out, including the ones that normally come from
`params_source.yaml` and the other default files, so editing those
files cannot change what the tests evaluate.


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
reference $\chi^2$ values, and rewrites the manifest. Review the printed
$\chi^2$ values against the old references before committing: they define
what every later test run compares against.
