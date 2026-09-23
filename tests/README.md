# Unit tests for the likelihoods

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

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [The CFASTPT vs FASTPT comparison](#cfastpt_fastpt)
    2. [Accuracy checks](#accuracy_checks)
    3. [Baryonic feedback accuracy checks](#baryon_accuracy_checks)
    4. [Baryonic feedback drift tests](#baryon_drift_tests)
3. [Appendix](#appendix)
    1. [FAQ: Do the tests keep their own data?](#frozen_copy)
    2. [FAQ: Why do the tests use their own data vectors?](#synthetic_vectors)
    3. [FAQ: How can maintainers refresh the snapshot?](#refreeze)

## Running the tests <a name="run_tests"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the tests of this project

    python -m pytest ./projects/desy1xplanck/tests

Without pytest:

    python -m unittest discover -s ./projects/desy1xplanck/tests -v

The tests change no project files. Each test prints a progress line
per model build and per evaluation, then a report with the
computed $\chi^2$, the stored reference, the difference, and the pass
limit.

A full run performs about 50 likelihood evaluations and takes a
few minutes. The test files force `OMP_NUM_THREADS=4` internally.

## The tests <a name="the_tests"></a>

The standard configurations get four tests each: a $\chi^2$ drift check
and a race-condition check, both in the NLA and in the TATT intrinsic-alignment
model. The TATT variants set

    IA_model: 1
    DES_A2_1: 0.05
    DES_BTA_1: 0.05
    DES_A2_2: -1.51541

The two checks and their pass limits:

| check | pass limit                                        | a failure means                    |
|-------|---------------------------------------------------|------------------------------------|
| $\Delta\chi^2$ | the recomputed $\chi^2$ must stay within 0.2 of the value stored in `frozen/reference_chi2.json` | code or data changed the numbers |
| race condition | the fiducial evaluated on its own vs evaluated again after nine other cosmologies; the two must agree within $10^{-4}$ | leftover state or an OpenMP race |

Everything the tests compare against lives under `frozen/`: one
snapshot of configurations, data, and reference values, captured
together when the references were generated and unchanged since. The
[Appendix](#appendix) explains how the snapshot is protected.

The test files and the configurations they cover:

| test | file | configuration | what it checks |
|---|---|---|---|
| 1 | `test_example1.py` | cosmic shear; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 2 | `test_example1.py` | cosmic shear; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 3 | `test_example1.py` | cosmic shear; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 4 | `test_example1.py` | cosmic shear; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 5 | `test_example2.py` | 6x2pt; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 6 | `test_example2.py` | 6x2pt; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 7 | `test_example2.py` | 6x2pt; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 8 | `test_example2.py` | 6x2pt; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 11 | `test_example2_2x2pt.py` | 2x2pt (`desy1xplanck.combo_2x2pt`: the 6x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 12 | `test_example2_2x2pt.py` | 2x2pt (`desy1xplanck.combo_2x2pt`: the 6x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 13 | `test_example2_2x2pt.py` | 2x2pt (`desy1xplanck.combo_2x2pt`: the 6x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 14 | `test_example2_2x2pt.py` | 2x2pt (`desy1xplanck.combo_2x2pt`: the 6x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing); IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 15 | `test_fastpt.py` | cosmic shear; IA modeling: TATT; the C cfastpt (`IA_code: 0`) vs the python FAST-PT package (`IA_code: 1`) at 30 fixed points (20 across the intrinsic-alignment prior plus a one-parameter-at-a-time family), cosmology at the fiducial | $\Delta\chi^2$ of the FAST-PT data vector against the cfastpt data vector at the same point; the cfastpt vector is that point's fiducial, so agreement means zero |

### The CFASTPT vs FASTPT comparison (`test_fastpt.py`, test 15) <a name="cfastpt_fastpt"></a>

Cosmolike computes the TATT perturbation-theory integrals with two
implementations: cfastpt, the C code built into the interface
(`IA_code: 0`), and the python FAST-PT package through the fastpt
theory block (`IA_code: 1`). Test 15 evaluates both at the same 30
fixed points: 20 drawn once across the intrinsic-alignment prior and
hard-coded, plus a one-parameter-at-a-time family that names the TATT
parameter driving a divergence (every other parameter stays at the
fiducial), each implementation in its own subprocess.

At every point the cfastpt data vector is the fiducial: the reported
quantity is the $\Delta\chi^2$ of the FAST-PT vector against it,
zero for identical vectors and quadratic in their difference. A
comparison against the shipped data vector would measure the slope
of the distance to the data instead of the numerics.

The FAST-PT side runs at the recommended minimum settings the
example yamls carry in their commented fastpt block, and the pass
limit is 0.2. A second FAST-PT evaluation at a doubled grid boost
repeats the measurement as an advisory. The point values, the
design, and the parameter attribution are shared with the lsst_y1
project (its tests/README.md carries the figures and the full
discussion); the convergence below is this project's own sweep:

| FAST-PT grid boost | max $\Delta\chi^2$ | median $\Delta\chi^2$ | cost per cosmology |
|---|---|---|---|
| 1 (shipped default) | 29.6 | 0.95 | 1.1 s |
| 20 | 1.52 | 0.049 | 1.3 s |
| 40 | 0.47 | 0.015 | 1.7 s |
| 80 (recommended minimum) | 0.132 | 0.0042 | 2.6 s |
| 160 | 0.036 | 0.0012 | 3.0 s |

> [!Warning]
> Do not lower the fastpt `accuracyboost` below 80 in a
> TATT analysis with `IA_code: 1`: the tidal-torquing and
> $b_{\rm TA}$ convolution terms are under-resolved at the shipped
> grid. Production analyses use cfastpt (`IA_code: 0`), the
> converged and faster reference.

#### Running the comparison <a name="run_cfastpt_fastpt"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the comparison at the default camb/cosmolike
settings

    python -m pytest ./projects/desy1xplanck/tests/test_fastpt.py

**Step :three:**: repeat it at the pushed camb/cosmolike settings

    python -m pytest ./projects/desy1xplanck/tests/test_fastpt.py --high=1

> [!NOTE]
> `--high=1`: applies the pushed camb/cosmolike settings of the
> accuracy checks to every block of test 15 (the other tests do not
> read it). The full comparison is both invocations.


### Accuracy checks (`test_accuracy.py`, A1-A6) <a name="accuracy_checks"></a>

We change one accuracy parameter at a time on the 6x2pt NLA
configuration, then checks A1-A6 re-evaluate cosmic shear, 6x2pt,
and 2x2pt, with NLA and TATT, with every setting pushed far beyond
the defaults at once. The scan keeps an `accuracyboost: 5` entry as
a deliberate stress test: in this project it breaks the 6x2pt
integration tables and produces a $\Delta\chi^2$ of +27, so it
stays out of the raised-at-once set below.

| setting | raised to | what it controls |
|---------|-----------|------------------|
| `accuracyboost` (cosmolike) | 2 | sizes of cosmolike's internal lookup tables, including the dyadic z grid of the power-spectrum tables |
| `integration_accuracy` (cosmolike) | 10 | extra refinement passes of cosmolike's numerical integrals |
| `lmax` (cosmolike) | 200000 | highest multipole of the internal harmonic-space $C_\ell$ tables that cosmolike transforms into the real-space correlation functions; arcminute scales need very high $\ell$ |
| `kmax_boltzmann` (cosmolike) | 40 | the k cutoff of the power spectrum the likelihood requests from CAMB |
| `AccuracyBoost` (CAMB) | 2 | CAMB's overall accuracy multiplier: denser sampling in every internal CAMB grid, the most expensive setting |
| `k_per_logint` (CAMB) | 50 | k samples CAMB computes per logarithmic interval of the transfer functions |
| `kmax` (CAMB) | 50 | highest k of CAMB's matter power spectrum; one physical cutoff with `kmax_boltzmann`, seen from the CAMB side |

`accuracyboost` refines a nested z grid in the power-spectrum
tables: every coarser grid's nodes are a subset of every finer
grid's, so a higher boost tightens the same interpolation instead of
moving the nodes (the construction is commented in
`likelihood/_cosmolike_prototype_base.py`).

When several settings move the $\chi^2$, settle them in cost order:
raise cosmolike `accuracyboost` first (cheap), then CAMB
`k_per_logint`, and CAMB `AccuracyBoost` last (expensive at run
time, and able to masquerade for the cheap settings).
`kmax_boltzmann` and CAMB `kmax` are one physical cutoff seen from
two sides; move them together.

Each check reports the $\Delta\chi^2$ between the high-accuracy and
the default evaluations: the numerical error of the default
settings. No pass/fail.

> [!NOTE]
> High-accuracy evaluations take minutes.

#### Running Accuracy checks <a name="run_accuracy"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the accuracy checks on their own

    python -m pytest ./projects/desy1xplanck/tests/test_accuracy.py

To run every other test while skipping these:

    python -m pytest ./projects/desy1xplanck/tests --ignore ./projects/desy1xplanck/tests/test_accuracy.py

### Baryonic feedback accuracy checks (`test_accuracy_baryons.py`, BF1-BF7) <a name="baryon_accuracy_checks"></a>

The file `test_accuracy_baryons.py` repeats the default-versus-high
accuracy comparison with the `bfmt` theory block switched on: one
advisory check per feedback method (the three SP(k) fb relations,
BCEmu, Flamingo, BACCOemu, and BCemu2025), at a fixed parameter
point per method. Each check creates its data vector on the fly, by
the same mechanism as the N-random-models check: the
default-settings model writes its own theory vector during
evaluation, that vector becomes the data of a temporary dataset, and
the pushed-settings model evaluates at the same point against it.
The fiducial $\chi^2$ is therefore zero by construction, nothing is
stored in the snapshot, and the single reported number,
$\Delta\chi^2$, is a pure numerics response. The check BF0
additionally runs the one-setting-at-a-time scan with the Akino
SP(k) feedback on, so a large delta names the setting causing it.

Every checked configuration is measurable by construction. The
BACCOemu check evaluates with `omegab: 0.049`, inside that
emulator's baryon-density training box, whose floor sits exactly
above the fiducial `omegab: 0.04`; and the double-power-law point is
chosen to keep the baryon fraction inside SP(k)'s calibrated band
over the full redshift grid.

#### Running the baryonic feedback checks <a name="run_baryon_accuracy"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the baryonic feedback checks of this project

    python -m pytest ./projects/desy1xplanck/tests/test_accuracy_baryons.py

### Baryonic feedback drift tests (`test_baryons.py`, BD1-BD7) <a name="baryon_drift_tests"></a>

The file `test_baryons.py` pins the feedback pipeline against change
over time, one test per method. Each method's default-settings
theory prediction was stored at freeze time
(`generate_frozen_reference.py --baryons`), and the test evaluates
today's prediction against that stored vector: zero at freeze time
by construction, so a $\chi^2$ above the tolerance means cosmolike
or the `bfmt` theory block changed its prediction since the freeze.
These tests complement the accuracy checks above: the accuracy
checks regenerate their vector on the fly per run, so they measure
the numerical settings and can never see drift; the drift tests hold
the frozen vector still, so they measure drift and nothing else.

#### Running the baryonic feedback drift tests <a name="run_baryon_drift"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the drift tests of this project

    python -m pytest ./projects/desy1xplanck/tests/test_baryons.py

# Appendix <a name="appendix"></a>

## :interrobang: FAQ: Do the tests keep their own data? <a name="frozen_copy"></a>

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
changes when any byte changes) of every file under `frozen/`. Each test
verifies the manifest first and refuses to run when a file under `frozen/` was
edited, naming the file. The result: users may change the live data
and examples freely, and nobody can quietly edit the snapshot
either.

## :interrobang: FAQ: Why do the tests use their own data vectors? <a name="synthetic_vectors"></a>

This project's shipped data vector is real data, and the example
cosmology is not its best fit, so the $\chi^2$ there sits far from the
minimum, where it responds linearly to tiny numerical changes. Every
variant therefore evaluates against a data vector generated at the
fiducial point when the snapshot was created:
`frozen/data/synthetic_desy1xplanck.dataset`
(default NLA model) for the NLA tests and
`frozen/data/tatt_desy1xplanck.dataset` (TATT model) for the TATT tests.
Both come from the 6x2pt model, whose full-length vector
serves every probe; at its own minimum the $\chi^2$ response is quadratic
and the drift and accuracy numbers stay meaningful.

## :interrobang: FAQ: How can maintainers refresh the snapshot? <a name="refreeze"></a>

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rebuild the snapshot

    python ./projects/desy1xplanck/tests/generate_frozen_reference.py --overwrite

It rebuilds `frozen/` from the current project, prints the four new
reference $\chi^2$ values, and rewrites the manifest. Review the printed
$\chi^2$ values against the old references before committing: they define
what every later test run compares against.
