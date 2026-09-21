"""Shared harness for the desy1xplanck unit tests.

The tests answer two questions about the desy1xplanck likelihoods:

  1. Does the chi2 at a fixed reference point still match the value
     recorded when the tests were created (tests 1, 3, 5, 7)?
  2. Does evaluating several cosmologies in a row on one model change
     the answer for a point, compared to evaluating that point alone
     (tests 2, 4, 6, 8)? Cosmolike keeps internal state in C between
     evaluations, and OpenMP splits loops across threads; a bug in
     either would make the 10th evaluation of a sequence differ from a
     fresh evaluation of the same point. That class of bug is called a
     race condition or a state leak.

Everything a test evaluates is FROZEN: stored under tests/frozen/ and
pinned by a SHA-256 hash (a 64-character fingerprint that changes when
any byte of the file changes) in tests/manifest_sha256.json. The live
project configuration is never read, so a user can edit the examples,
the likelihood default yaml files, or ../data without touching these
tests. The frozen state has three parts:

  - frozen/frozen_config_example{1,2}.py: one auto-generated module per
    example holding (a) the complete cobaya configuration as a yaml
    string, with every likelihood option and every parameter written
    out, including the ones that normally come from the likelihood
    default files (cosmic_shear.yaml, combo_6x2pt.yaml,
    params_source.yaml, params_lens.yaml), and (b) the exact
    sampled-parameter point the reference chi2 was evaluated at.
    Because every default is materialized in the frozen copy, a later
    edit to a live default file is shadowed and cannot reach the test.
  - frozen/data/: the tests' own copy of the data vectors, covariance,
    n(z), and masks.
  - frozen/EXAMPLE_EVALUATE{1,2}.yaml: snapshots of the example yaml
    files at freeze time, kept only so a human can see what
    changed in the live examples since the freeze; no test reads
    them.

Every model build runs in its own worker subprocess: the public
single_model_chi2 and ten_in_a_row_chi2 spawn a fresh python that
imports this module, evaluates, and hands the numbers back through a
temporary json file, while its progress lines stream to the same
terminal. In this project both examples share one data set, so the
isolation is preventive rather than required: in a project whose
examples use different data-set dimensions (desy1xplanck), the cosmolike
C layer aborts a process that initializes both, and every project
keeps one architecture so they all behave identically.

Every test first verifies the manifest and refuses to run when any
frozen file changed. Refreshing the frozen state is a deliberate
maintainer action: generate_frozen_reference.py --overwrite.

To run the tests: activate the cocoa conda environment, source
start_cocoa.sh from the Cocoa/ folder (this exports ROOTDIR), then

    python -m pytest ./projects/desy1xplanck/tests
"""

import hashlib
import json
import os

# Everything the tests read or write lives relative to this folder, so
# the suite works no matter which directory pytest is launched from.
# __file__ is this module's own file path; abspath + dirname reduce it
# to the tests/ folder.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

# The race tests must run multi-threaded: with one thread there is no
# thread scheduling, so an OpenMP race could never show up.
REQUIRED_OMP_THREADS = "4"

# Tests 1, 3, 5, 7: |chi2(now) - chi2(frozen reference)| must stay
# below this. The bound tolerates compiler and library-version noise
# but catches a real physics change.
CHI2_TOLERANCE = 0.2

# Tests 2, 4, 6, 8: |chi2(10th of a row) - chi2(fresh model)|. The two
# numbers come from the same code on the same inputs, so only float
# noise is allowed; a state leak produces a much larger shift.
RACE_TOLERANCE = 1.0e-4

# The TATT (Tidal Alignment and Tidal Torquing, an intrinsic-alignment
# model with tidal second-order terms) tests replace these values in
# the frozen point. In the NLA reference point A2 and BTA are zero, so
# the nonzero values here make the TATT reference genuinely exercise
# the second-order terms.
TATT_POINT = {
    "DES_A2_1": 0.05,
    "DES_BTA_1": 0.05,
    "DES_A2_2": -1.51541,
}

# The two frozen configurations. "likelihood" is the cobaya component
# name, needed to reach that block inside the loaded info dictionary;
# "provenance" names the human-readable snapshot (never loaded).
# The TATT variants evaluate against a data vector GENERATED WITH
# TATT at the fiducial point. Reason: against an NLA-based vector the
# TATT chi2 sits away from its minimum, where it responds linearly
# (not quadratically) to tiny numerical changes: harmless
# rounding-level shifts would then eat much of the 0.2 chi2 band the
# reference tests allow. Both examples share one data set, so a single full-length
# vector generated from the example2 TATT model serves every
# configuration (the other probes' masks select their sections).
TATT_GENERATORS = {
    "tatt_desy1xplanck.dataset": "example2",
}

# This project's shipped data_file is REAL data, and the example
# cosmology is not its best fit: the chi2 sits far from the minimum
# (hundreds to thousands), where it responds LINEARLY to tiny theory
# changes. A chi2 comparison evaluated there reports alarming
# shifts that say nothing about the numerics near a fit. The NLA
# variants therefore evaluate against a SYNTHETIC data vector,
# generated with the default (NLA) model at the fiducial point from
# the example2 model during the freeze, exactly like the TATT vector:
# at its own minimum the chi2 response is quadratic and stable.
NLA_DATASET = "synthetic_desy1xplanck.dataset"

# Every generated vector: {".dataset" filename: (source example, TATT?)}.
# One full-length vector per IA model, generated from the example2
# model, serves every configuration (the other probes' masks select
# their sections).
SYNTHETIC_VECTORS = {
    NLA_DATASET: ("example2", False),
    "tatt_desy1xplanck.dataset": ("example2", True),
}

# High-accuracy settings for the accuracy advisory checks
# (test_accuracy.py): the same physics evaluated with the numerical
# knobs pushed far beyond the defaults.
HIGH_ACCURACY_LIKELIHOOD = {
    # boost 3 is the highest value that stays healthy in every project
    # scanned (desy1xplanck breaks down above it), so the all-knobs
    # check compares the default against 3; the one-at-a-time scan
    # keeps 5 as a deliberate stress knob
    "accuracyboost": 3.0,       # default 1.0
    "integration_accuracy": 10,  # default 0
    "lmax": 200000,             # default 50000-75000
    "kmax_boltzmann": 40.0,     # default 5.0-7.5
}
HIGH_ACCURACY_CAMB_EXTRA_ARGS = {
    "halofit_version": "takahashi",
    "AccuracyBoost": 2.0,       # default 1.05
    "dark_energy_model": "ppf",
    "accurate_massive_neutrino_transfers": False,
    "k_per_logint": 50,         # default 10
    "kmax": 50.0,               # default 5.0-7.5
}

# The one-at-a-time scan of test_accuracy.py: each entry is (label,
# likelihood overrides, camb extra_args overrides), evaluated alone on
# the example2 NLA configuration before the all-knobs checks, so a
# large all-knobs delta can be attributed to the knob causing it. The
# accuracyboost=5 entry is a stress knob: it exceeds what measuring
# the default numerics needs, and it is kept because it exposed an
# interface breakdown (a suspected fixed-size table) in desy1xplanck.
# Investigation order when several knobs move the chi2: raise the
# cosmolike accuracyboost first (cheap), then camb k_per_logint, and
# only then camb AccuracyBoost (expensive at run time): an apparent
# CAMB sensitivity can masquerade as unresolved cosmolike-side
# resolution, so the cheap knobs must be settled before the expensive
# one is blamed. kmax_boltzmann and camb kmax are one physical cutoff
# seen from the two sides, so the scan moves them together.
ACCURACY_KNOBS = [
    ("accuracyboost -> 3", {"accuracyboost": 3.0}, {}),
    ("accuracyboost -> 5 (stress)", {"accuracyboost": 5.0}, {}),
    ("integration_accuracy -> 10", {"integration_accuracy": 10}, {}),
    ("lmax -> 200000", {"lmax": 200000}, {}),
    ("kmax_boltzmann -> 40 + camb kmax -> 50",
     {"kmax_boltzmann": 40.0}, {"kmax": 50.0}),
    ("camb AccuracyBoost -> 2", {}, {"AccuracyBoost": 2.0}),
    ("camb k_per_logint -> 50", {}, {"k_per_logint": 50}),
]

EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "desy1xplanck.cosmic_shear",
        "tatt_dataset": "tatt_desy1xplanck.dataset",
    },
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "desy1xplanck.combo_6x2pt",
        "tatt_dataset": "tatt_desy1xplanck.dataset",
    },
    "example2_2x2pt": {
        "frozen_module": "frozen_config_example2_2x2pt.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "source_likelihood": "desy1xplanck.combo_6x2pt",
        "likelihood": "desy1xplanck.combo_2x2pt",
        "tatt_dataset": "tatt_desy1xplanck.dataset",
    },
}

# The nine cosmologies evaluated before the fiducial point in a race
# test. Each entry replaces the named parameters in the frozen point.
# They stay inside the priors of the frozen configuration (an
# out-of-prior point would evaluate to -inf and abort the test), and
# they change the chi2 by orders of magnitude, so state leaked from
# any of them would visibly move the final fiducial evaluation.
RACE_PERTURBATIONS = [
    {"As_1e9": 1.95},
    {"As_1e9": 2.25},
    {"omegam": 0.28},
    {"omegam": 0.33},
    {"H0": 64.0},
    {"H0": 71.0},
    {"ns": 0.95},
    {"w": -1.1, "w0pwa": -1.1},
    {"omegab": 0.052, "mnu": 0.15},
]


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
def require_cocoa_environment():
    """Refuse to run outside a started Cocoa shell, then move to ROOTDIR.

    start_cocoa.sh exports ROOTDIR (the absolute path of the Cocoa/
    folder) and prepares the library paths the compiled cosmolike
    interface needs. Without it, importing the likelihood would fail
    with a confusing linker error, so this check turns that failure
    into an instruction. The chdir matters because component paths in
    the frozen configuration (for example CAMB's
    ./external_modules/code/CAMB) are relative to ROOTDIR.

    Returns:
      nothing; on success the process working directory is ROOTDIR.

    Raises:
      RuntimeError telling the user to activate the cocoa environment
      and source start_cocoa.sh when ROOTDIR is not exported.
    """
    if "ROOTDIR" not in os.environ:
        raise RuntimeError(
            "ROOTDIR is not set. Activate the cocoa conda environment and run "
            "`source start_cocoa.sh` from the Cocoa/ folder before running "
            "these tests."
        )
    os.chdir(os.environ["ROOTDIR"])


def assert_omp_threads():
    """Refuse a race test that would not actually run multi-threaded.

    OpenMP reads OMP_NUM_THREADS once, when the compiled library is
    first loaded, so the value must be in the environment before any
    cobaya or cosmolike import. The test modules set it at their first
    line; this check catches a run that imported the stack some other
    way first (for example from an interactive session).

    Returns:
      nothing when OMP_NUM_THREADS equals REQUIRED_OMP_THREADS.

    Raises:
      RuntimeError naming the observed value and the required one.
    """
    # .get returns None when the variable is unset, where indexing
    # would raise a KeyError; None then fails the comparison below
    # with the same readable message. In that message, !r prints the
    # value as python source (None without quotes, a string with
    # them), telling unset apart from empty.
    observed = os.environ.get("OMP_NUM_THREADS")
    if observed != REQUIRED_OMP_THREADS:
        raise RuntimeError(
            f"OMP_NUM_THREADS={observed!r}; the race-condition tests require "
            f"OMP_NUM_THREADS={REQUIRED_OMP_THREADS} and it must be set "
            "before cobaya/cosmolike are imported."
        )


# -----------------------------------------------------------------------------
# Frozen-state integrity
# -----------------------------------------------------------------------------
def sha256_of(path):
    """Fingerprint one file with SHA-256.

    Arguments:
      path = absolute path of the file to hash.

    Returns:
      the 64-character lowercase hexadecimal SHA-256 digest of the
      file's bytes. Reading happens in 1 MiB blocks so the 80 MB
      covariance never sits in memory at once.
    """
    hasher = hashlib.sha256()
    # "rb" reads raw bytes; the with closes the file when the block
    # ends, even if reading raises
    with open(path, "rb") as f:
        # two-argument iter(callable, sentinel) calls the lambda
        # again and again until it returns b"" (end of file);
        # 1 << 20 is 2 to the 20th = 1 MiB, the size of each read
        for block in iter(lambda: f.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def compute_manifest():
    """Hash every file currently under tests/frozen/.

    __pycache__ folders and .pyc files are skipped: Python writes them
    as a side effect of importing the frozen modules, so hashing them
    would make the manifest fail after the first run. .DS_Store files
    (macOS Finder metadata) are skipped for the same reason.

    Returns:
      a dictionary {relative path: sha256 digest}, with paths relative
      to the tests/ folder using "/" separators, sorted by path so the
      manifest file is stable across platforms.
    """
    files = {}
    # os.walk visits frozen/ and every folder below it, handing back
    # (folder, subfolder names, file names) one folder at a time
    for base, dirs, names in os.walk(FROZEN_DIR):
        # the comprehension keeps every name except __pycache__, and
        # assigning to dirs[:] rewrites the list os.walk is holding
        # in place, so the walk never descends into the dropped folder
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(names):
            if name == ".DS_Store" or name.endswith(".pyc"):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, TESTS_DIR).replace(os.sep, "/")
            files[rel] = sha256_of(full)
    # sorted(files.items()) orders the (path, digest) pairs by path;
    # dict() rebuilds the table in that order (dicts remember
    # insertion order), so the manifest writes identically everywhere
    return dict(sorted(files.items()))


def verify_frozen():
    """Fail every test up front when the frozen state was edited.

    Compares the stored manifest with a fresh hash of tests/frozen/ in
    both directions, so an edited file (CHANGED), a deleted file
    (MISSING), and a new file (EXTRA) are all reported. This runs
    before any model is built: a tampered frozen state must not
    produce a plausible-looking chi2.

    Returns:
      nothing when every frozen file matches the manifest.

    Raises:
      AssertionError listing every mismatched path and pointing to
      generate_frozen_reference.py --overwrite for a deliberate
      refresh; AssertionError also when the manifest file itself is
      absent (the frozen state was never generated).
    """
    if not os.path.isfile(MANIFEST_FILE):
        raise AssertionError(
            "tests/manifest_sha256.json is missing; run "
            "generate_frozen_reference.py --overwrite to create the frozen "
            "test state."
        )
    # expected = the {relative path: sha256 digest} table written at
    # freeze time; it is the definition of "untouched". json.load
    # turns the file's JSON text back into nested dictionaries, and
    # the with closes the file once the block ends, error or not
    with open(MANIFEST_FILE) as f:
        expected = json.load(f)["files"]
    # actual = the same table computed from the files on disk right now
    # (compute_manifest walks frozen/ and fingerprints each file)
    actual = compute_manifest()
    # collect every discrepancy before raising: a report naming all
    # problem files at once beats failing on the first one
    problems = []
    # .items() hands out (path, digest) pairs, unpacked into the two
    # loop names
    for rel, digest in expected.items():
        if rel not in actual:
            # the manifest lists it but the file is gone from disk
            problems.append(f"MISSING  {rel}")
        elif actual[rel] != digest:
            # the file exists but at least one byte differs
            problems.append(f"CHANGED  {rel}")
    # both directions matter: a file ADDED to frozen/ is as suspicious
    # as an edited one, so the reverse scan runs too
    for rel in actual:
        if rel not in expected:
            problems.append(f"EXTRA    {rel}")
    if problems:
        # "\n  ".join(problems) glues the collected lines into one
        # indented block, one mismatch per line
        raise AssertionError(
            "Frozen test data does not match tests/manifest_sha256.json "
            "(someone edited the frozen copies; the tests refuse to run):\n  "
            + "\n  ".join(problems)
            + "\nIf the change is deliberate, regenerate with "
            "generate_frozen_reference.py --overwrite."
        )


def load_reference():
    """Read the frozen reference chi2 values.

    Returns:
      the dictionary stored in frozen/reference_chi2.json: one entry
      per configuration ("example1_nla", "example1_tatt",
      "example2_nla", "example2_tatt") plus a "_meta" entry recording
      when and how the references were generated. The file sits inside
      frozen/, so verify_frozen() also protects it from editing.
    """
    # json.load parses the file's JSON text back into the dictionary
    # json.dump wrote; the with closes the file on every exit
    with open(REFERENCE_FILE) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Terminal reports
# -----------------------------------------------------------------------------
def report_chi2_test(number, label, chi2, ref, tol):
    """Print one reference-comparison test as a readable block.

    A bare pytest PASSED does not say what was compared, so each test
    prints its own numbers: the freshly computed chi2, the frozen
    reference, their absolute difference, and the limit the assertion
    uses. flush=True makes the block appear immediately (pytest runs
    with -s, so nothing buffers it).

    Arguments:
      number = the test number (1-8) shown in the header.
      label  = one line naming the example, probe, and IA model.
      chi2   = the chi2 computed in this run.
      ref    = the frozen reference chi2.
      tol    = the pass limit on |chi2 - ref| (CHI2_TOLERANCE).

    Returns:
      |chi2 - ref|, the printed difference.
    """
    delta = abs(chi2 - ref)
    # in the f-string: '-' * 66 repeats the dash into a 66-character
    # rule, :.6f prints fixed six decimals, and the a-if-c-else-b at
    # the arrow picks the verdict word from the comparison
    print(f"""
{'-' * 66}
TEST {number}: {label}
  chi2 (this run)     = {chi2:.6f}
  frozen reference    = {ref:.6f}
  |delta chi2|        = {delta:.6f}   (limit: < {tol})
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_race_test(number, label, fresh, tenth, tol):
    """Print one race-condition test as a readable block.

    Arguments:
      number = the test number (1-8) shown in the header.
      label  = one line naming the example, probe, and IA model.
      fresh  = chi2 of the fiducial point evaluated first on the model.
      tenth  = chi2 of the same point evaluated as the 10th of a row.
      tol    = the pass limit on |tenth - fresh| (RACE_TOLERANCE).

    Returns:
      |tenth - fresh|, the printed difference.
    """
    delta = abs(tenth - fresh)
    # :.8f = eight fixed decimals; the 1e-4 race band needs more
    # digits than the six the reference blocks print
    print(f"""
{'-' * 66}
TEST {number}: {label}
  fresh-model chi2    = {fresh:.8f}
  10th of 10 in a row = {tenth:.8f}
  |delta chi2|        = {delta:.8f}   (limit: < {tol})
  OMP_NUM_THREADS     = {os.environ.get('OMP_NUM_THREADS')}
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_accuracy(label, chi2_high, default_ref):
    """Print one default-vs-high-accuracy check. Advisory only.

    The default-settings chi2 is the frozen reference (recorded at
    freeze time); the high-accuracy chi2 is computed in this run. The
    difference is the numerical error of the default settings at this
    point: there is no pass/fail because how much numerical error an
    analysis tolerates is a judgment call, not a fixed bound.

    Arguments:
      label       = one line naming the probe and IA model.
      chi2_high   = chi2 with HIGH_ACCURACY settings, this run.
      default_ref = the frozen default-settings reference chi2.

    Returns:
      chi2_high - default_ref, the printed difference.
    """
    delta = chi2_high - default_ref
    # the + in {delta:+.6f} forces a sign: the direction of the
    # numerical drift matters as much as its size
    print(f"""
{'-' * 66}
ACCURACY: {label}
  chi2 (high accuracy)      = {chi2_high:.6f}
  chi2 (default, frozen)    = {default_ref:.6f}
  delta chi2 (high-default) = {delta:+.6f}
{'-' * 66}""", flush=True)
    return delta


def report_knob(label, chi2, default_ref):
    """Print one entry of the one-knob-at-a-time scan. Advisory only.

    Arguments:
      label       = the ACCURACY_KNOBS entry evaluated.
      chi2        = chi2 with only that knob changed, this run.
      default_ref = the frozen default-settings reference chi2.

    Returns:
      chi2 - default_ref, the printed difference.
    """
    delta = chi2 - default_ref
    # :30s pads the label to 30 characters so the scan lines land in
    # columns; :12.6f = six decimals in a 12-wide field, and the +
    # variant forces a sign on the delta
    print(f"  KNOB {label:30s} chi2 = {chi2:12.6f}  "
          f"delta = {delta:+12.6f}", flush=True)
    return delta


# -----------------------------------------------------------------------------
# Model construction and evaluation
# -----------------------------------------------------------------------------
# cobaya and numpy are imported inside the functions below, not at the
# top of this module. The reason is OpenMP: OMP_NUM_THREADS must be in
# the environment before the compiled libraries load, and it is the
# TEST modules that set it, on their first line, before importing this
# module's callers.
def _frozen_module(example):
    """Load one frozen configuration module from its file path.

    importlib is used instead of a plain import statement because the
    frozen modules live inside frozen/, which is data, not a package:
    it has no __init__.py and is never on sys.path. Loading by path
    also guarantees the file that verify_frozen() hashed is exactly
    the file being executed.

    Arguments:
      example = "example1" or "example2" (a key of EXAMPLES).

    Returns:
      the loaded module, carrying the attributes `yaml_string` (the
      complete configuration) and `point` (the frozen evaluation
      point).
    """
    import importlib.util

    path = os.path.join(FROZEN_DIR, EXAMPLES[example]["frozen_module"])
    # the three importlib steps mirror what `import` does under the
    # hood: build a loading recipe (spec) for this one file, make an
    # empty module from it, then run the file's code inside the
    # module so its top-level assignments become module attributes
    spec = importlib.util.spec_from_file_location(f"frozen_{example}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_info(example, tatt, high_accuracy=False,
                     overrides=None):
    """Build the cobaya input dictionary for one frozen configuration.

    Starts from the frozen module's yaml string and applies the only
    three run-time adjustments the tests need:

      - the likelihood `path` is pointed at the absolute location of
        frozen/data on this machine (the frozen string stores the
        ROOTDIR-relative form, which would also resolve, but the
        absolute path is independent of the working directory);
      - `IA_model` selects the intrinsic-alignment model: 0 keeps NLA,
        1 selects TATT;
      - cobaya's log level is raised to WARNING (debug: 30) so the
        component-loading chatter does not bury the test reports.

    Arguments:
      example = a key of EXAMPLES.
      tatt    = True selects the TATT IA model, False keeps NLA; the
                data_file then switches to the configuration's
                TATT-generated dataset (see TATT_GENERATORS).
      high_accuracy = True applies HIGH_ACCURACY_LIKELIHOOD and
                HIGH_ACCURACY_CAMB_EXTRA_ARGS on top of the frozen
                configuration (accuracy advisory checks only).

    Returns:
      the input dictionary ready for cobaya's get_model.
    """
    from cobaya.yaml import yaml_load

    cfg = EXAMPLES[example]
    # _frozen_module loads frozen/<frozen_module>.py by path and hands
    # back its yaml_string attribute: the complete configuration with
    # every option and parameter written out at freeze time
    info = yaml_load(_frozen_module(example).yaml_string)
    # the tests drive the model directly, so a sampler or output block
    # left in the info would only confuse cobaya. pop(key, None)
    # removes the key when present and does nothing (no error) when
    # the frozen configuration never had it
    info.pop("sampler", None)
    info.pop("output", None)
    # log level WARNING (30): component-loading chatter would bury the
    # test reports
    info["debug"] = 30
    info["timing"] = False
    likelihood_block = info["likelihood"][cfg["likelihood"]]
    # the frozen string stores the ROOTDIR-relative data path; the
    # absolute path is independent of the working directory
    likelihood_block["path"] = os.path.join(FROZEN_DIR, "data")
    # tests must not write files as a side effect: this project's
    # example1 ships print_datavector: True aimed at chains/, which
    # does not exist in a fresh clone (the TATT-vector generator
    # re-enables printing deliberately, into frozen/data)
    likelihood_block["print_datavector"] = False
    # intrinsic-alignment model selection: 0 = NLA, 1 = TATT (the
    # ternary `1 if tatt else 0` reads: 1 when tatt is True, else 0)
    likelihood_block["IA_model"] = 1 if tatt else 0
    if tatt:
        # TATT evaluates against its own generated data vector so the
        # chi2 sits at a minimum (see the TATT_GENERATORS comment)
        likelihood_block["data_file"] = cfg["tatt_dataset"]
    else:
        # NLA does the same against the synthetic NLA vector: the
        # shipped data_file is real data and the fiducial point is far
        # from its minimum (see the NLA_DATASET comment)
        likelihood_block["data_file"] = NLA_DATASET
    if high_accuracy:
        # .update copies every entry of the high-accuracy table into
        # the block, overwriting the frozen value of any shared key
        likelihood_block.update(HIGH_ACCURACY_LIKELIHOOD)
        info["theory"]["camb"]["extra_args"].update(
            HIGH_ACCURACY_CAMB_EXTRA_ARGS)
    if overrides is not None:
        # one knob at a time (an ACCURACY_KNOBS entry): the same
        # mechanism as high_accuracy, restricted to a single setting
        like_over, camb_over = overrides
        likelihood_block.update(like_over)
        info["theory"]["camb"]["extra_args"].update(camb_over)
    if tatt:
        # a TATT parameter can be SAMPLED in the frozen configuration
        # (it has a prior; build_point then sets its value in the
        # evaluation point) or FIXED (a plain value; it must be
        # replaced here, before the model is built, because a fixed
        # parameter cannot change per evaluation)
        # .items() hands out (name, value) pairs, unpacked into the
        # two loop names
        for name, value in TATT_POINT.items():
            # .get returns None when the configuration carries no
            # parameter of that name at all
            block = info["params"].get(name)
            if block is None:
                raise ValueError(
                    f"TATT parameter {name} is not in the frozen "
                    "configuration")
            # `"prior" not in block` asks whether the dict lacks
            # that key: a parameter block without a prior is fixed
            if isinstance(block, dict) and "prior" not in block:
                block["value"] = value
    return info


def make_model(info):
    """Build a cobaya model (theory plus likelihood, ready to evaluate).

    Arguments:
      info = a cobaya input dictionary from load_frozen_info.

    Returns:
      the cobaya Model. Building one loads CAMB and the compiled
      cosmolike interface and reads the frozen data files, which takes
      a few seconds; the callers print a progress line first.
    """
    from cobaya.model import get_model

    return get_model(info)


def load_frozen_point(example):
    """Read the frozen evaluation point of one example.

    Arguments:
      example = "example1" or "example2" (a key of EXAMPLES).

    Returns:
      a fresh {parameter name: value} dictionary (copied, so a caller
      may modify it without affecting later calls).
    """
    # dict(...) builds a NEW dictionary with the same entries: the
    # caller may edit the copy without touching the frozen module
    return dict(_frozen_module(example).point)


def build_point(model, example, tatt):
    """Assemble the exact point a test evaluates, with a safety check.

    The frozen point must cover the model's sampled parameters one to
    one. When likelihood or theory code changes its parameter set (a
    new nuisance parameter appears, or one is removed), evaluating
    would either fail cryptically or silently pick up a new default,
    so the mismatch is reported here by name instead.

    Arguments:
      model   = the cobaya Model the point will be evaluated on.
      example = "example1" or "example2" (a key of EXAMPLES).
      tatt    = True replaces the TATT_POINT values (nonzero A2/BTA)
                in the frozen point; False evaluates it unchanged.

    Returns:
      a {parameter name: value} dictionary covering every sampled
      parameter of the model.

    Raises:
      AssertionError listing the parameters that appeared or vanished
      when the model's sampled set differs from the frozen point;
      ValueError when a TATT parameter is not sampled by the model.
    """
    # load_frozen_point returns a copy of the frozen module's point:
    # the exact {parameter: value} table the references were computed at
    point = load_frozen_point(example)
    # sampled = the parameters THIS model, built from today's code,
    # expects to receive; the frozen point must cover them exactly.
    # set() collects the names for comparison by content, order
    # ignored, and set(point) is the set of the dictionary's KEYS
    sampled = set(model.parameterization.sampled_params())
    if sampled != set(point):
        # sampled - set(point) is set difference: the names in the
        # first set only, here the parameters the code gained; the
        # mirrored expression lists the ones it lost
        raise AssertionError(
            "sampled-parameter set differs from the frozen point (the "
            "likelihood/theory code changed its parameters):\n"
            f"  new since freeze: {sorted(sampled - set(point))}\n"
            f"  gone since freeze: {sorted(set(point) - sampled)}"
        )
    if tatt:
        # only SAMPLED TATT parameters appear in the point; the fixed
        # ones were already replaced inside the configuration by
        # load_frozen_info (which also catches unknown names)
        for name, value in TATT_POINT.items():
            # `name in point` tests the dictionary's KEYS
            if name in point:
                point[name] = value
    return point


def evaluate_chi2(model, point):
    """Evaluate one point and return the likelihood chi2.

    cached=False forces a full recomputation: the race tests evaluate
    the same point twice on one model, and letting cobaya return a
    cached value would compare a number with itself.

    Arguments:
      model = the cobaya Model to evaluate on.
      point = {parameter name: value} covering the sampled parameters.

    Returns:
      chi2 = -2 ln L of the single likelihood, as a plain float.

    Raises:
      RuntimeError when the model holds more than one likelihood
      (the -2*loglikes[0] extraction would then be ambiguous);
      AssertionError when the chi2 is not finite, which is how an
      out-of-prior or rejected point shows up.
    """
    import numpy as np

    # logposterior runs the full pipeline (theory + likelihood) at the
    # point; cached=False forces recomputation (see docstring)
    posterior = model.logposterior(point, cached=False)
    # loglikes = one ln L per likelihood component, in model order
    if len(posterior.loglikes) != 1:
        raise RuntimeError(
            f"expected exactly one likelihood: {posterior.loglikes}")
    chi2 = -2.0 * posterior.loglikes[0]
    if not np.isfinite(chi2):
        raise AssertionError(f"non-finite chi2 at point {point}")
    return float(chi2)


def _single_model_chi2_impl(example, tatt, high_accuracy=False,
                            knob=None):
    """In-process body of single_model_chi2 (worker side).

    Runs inside the worker subprocess only: building a model here,
    next to a model of different dimensions, would abort the process
    (see the module docstring).

    Arguments:
      example = a key of EXAMPLES.
      tatt    = True evaluates the TATT variant, False the NLA one.
      high_accuracy = True evaluates with the pushed numerical
                settings (see load_frozen_info).

    Returns:
      the chi2 as a float.
    """
    # the ternary a if c else b picks "TATT" when tatt is True and
    # "NLA" otherwise; the label only feeds the progress line
    ia_label = "TATT" if tatt else "NLA"
    if high_accuracy:
        ia_label += ", high accuracy"
    overrides = None
    if knob is not None:
        # knob = a label from ACCURACY_KNOBS; the comprehension keeps
        # the entries whose label matches, so a right name yields a
        # one-entry list and a wrong one an empty list
        matches = [k for k in ACCURACY_KNOBS if k[0] == knob]
        if len(matches) != 1:
            raise ValueError(f"unknown accuracy knob {knob!r}")
        overrides = (matches[0][1], matches[0][2])
        ia_label += f", knob: {knob}"
    print(f"  building model ({example}, {ia_label}) ...", flush=True)
    # load_frozen_info returns the frozen configuration dictionary with
    # the run-time adjustments applied; make_model turns it into an
    # evaluable cobaya Model (loads CAMB and the cosmolike interface)
    info = load_frozen_info(example, tatt, high_accuracy=high_accuracy,
                            overrides=overrides)
    model = make_model(info)
    # build_point returns the frozen evaluation point after checking
    # that the point and the model name the same sampled parameters:
    # if the likelihood or theory code gained or lost a sampled
    # parameter since the freeze, the mismatch is reported by name
    # instead of failing deep inside cobaya
    point = build_point(model, example, tatt)
    print("  evaluating the fiducial point ...", flush=True)
    return evaluate_chi2(model, point)


def _ten_in_a_row_impl(example, tatt):
    """In-process body of ten_in_a_row_chi2 (worker side).

    Race check: the fiducial evaluated fresh and as 10th of a row.

    On ONE model instance, in order: the fiducial point (the fresh
    value), then the nine RACE_PERTURBATIONS cosmologies, then the
    fiducial again as the 10th point of the row. State leaked between
    evaluations, or an OpenMP race under REQUIRED_OMP_THREADS threads,
    shifts the second fiducial value away from the first; correct
    code reproduces it to float noise. Each evaluation prints its
    chi2, so a stuck or slow run is visible line by line.

    Arguments:
      example = "example1" or "example2" (a key of EXAMPLES).
      tatt    = True runs the TATT variant, False the NLA one.

    Returns:
      (fresh, tenth): chi2 of the first fiducial evaluation and chi2
      of the fiducial as the 10th point of the row, both floats.
    """
    # ternary: "TATT" when tatt is True, "NLA" otherwise
    ia_label = "TATT" if tatt else "NLA"
    print(f"  building model ({example}, {ia_label}) ...", flush=True)
    # one model instance for the whole sequence: sharing the instance
    # is the point, since leaked state lives inside it
    info = load_frozen_info(example, tatt)
    model = make_model(info)
    point = build_point(model, example, tatt)
    # the fresh value: the fiducial evaluated before anything else
    # touched this model instance
    fresh = evaluate_chi2(model, point)
    # the format spec :.8f prints fixed-point with eight decimals,
    # enough to see a float-noise difference against the 1e-4 band
    print(f"  fresh model, fiducial point:  chi2 = {fresh:.8f}", flush=True)
    # enumerate pairs each perturbation with a counter; start=1 makes
    # the printed rows read 1..9 instead of 0..8
    for i, perturbation in enumerate(RACE_PERTURBATIONS, start=1):
        # {**point, **perturbation} builds a NEW dict: point's
        # entries first, then the perturbation's on top of any
        # shared key; point itself stays untouched for the final
        # fiducial evaluation
        chi2 = evaluate_chi2(model, {**point, **perturbation})
        # join feeds on a generator: one "name=value" string per
        # changed parameter, glued with ", " between them
        changed = ", ".join(f"{k}={v}" for k, v in perturbation.items())
        # {i:2d} pads the counter to two characters so the rows line
        # up; :.4f prints four fixed decimals
        print(f"  row {i:2d}/10 ({changed}):  chi2 = {chi2:.4f}", flush=True)
    tenth = evaluate_chi2(model, point)
    print(f"  row 10/10 (fiducial again):  chi2 = {tenth:.8f}", flush=True)
    return fresh, tenth


# -----------------------------------------------------------------------------
# Worker-subprocess isolation
# -----------------------------------------------------------------------------
# The absolute path of this file: the worker driver imports the module
# by path, so the child executes exactly the code the parent runs.
_THIS_FILE = os.path.abspath(__file__)

# One flag separates the two roles: the parent process (pytest or the
# generator) spawns workers; a process carrying this environment
# variable IS a worker and runs the physics in-process.
_WORKER_FLAG = "COCOA_TESTS_WORKER"

# The driver handed to `python -c` inside the worker: load this module
# from its file path and call _worker with the six command-line
# arguments (function name, example, two booleans, a reserved slot,
# and the result path). Adjacent string literals glue into one
# string, each piece ending in its own \n, and *sys.argv[2:8]
# spreads that slice of the argument list into six call arguments.
_WORKER_DRIVER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location("
    "'cocoa_test_utils_worker', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "module._worker(*sys.argv[2:8])\n"
)


def _worker(function, example, tatt, high_accuracy, knob, result_path):
    """Worker-side entry: run one evaluation and save the numbers.

    Arguments:
      function      = "single" (one chi2) or "race" (fresh, tenth).
      example       = a key of EXAMPLES.
      tatt          = "1" for the TATT variant, "0" for NLA.
      high_accuracy = "1" for the pushed numerical settings, "0" not.
      knob          = an ACCURACY_KNOBS label evaluated alone, or the
                      empty string for none.
      result_path   = file the result is written into as json; the
                      parent reads it back. Progress prints go to the
                      inherited stdout, so the terminal streams them.

    Returns:
      nothing; the result lands in result_path.
    """
    require_cocoa_environment()
    # command-line arguments arrive as strings; comparing to "1"
    # turns each flag back into the True/False it encodes
    tatt = tatt == "1"
    high_accuracy = high_accuracy == "1"
    if function == "single":
        # `knob or None` turns the empty string back into None: or
        # hands back its second operand when the first counts false
        value = _single_model_chi2_impl(example, tatt,
                                        high_accuracy=high_accuracy,
                                        knob=knob or None)
    else:
        # list() turns the (fresh, tenth) pair into a list, the form
        # json.dump can store
        value = list(_ten_in_a_row_impl(example, tatt))
    with open(result_path, "w") as f:
        # json.dump writes the value as JSON text; the parent's
        # json.load hands the same number(s) back
        json.dump(value, f)


def _run_isolated(function, example, tatt, high_accuracy=False, knob=None):
    """Spawn one worker subprocess and hand back its result.

    Arguments:
      function      = "single" or "race" (see _worker).
      example       = a key of EXAMPLES.
      tatt          = True for the TATT variant.
      high_accuracy = True for the pushed numerical settings.

    Returns:
      the json value the worker wrote: a float for "single", a
      two-element list [fresh, tenth] for "race".

    Raises:
      RuntimeError naming the configuration when the worker dies
      without writing a result (the cosmolike C layer aborts the
      process on an internal inconsistency instead of raising).
    """
    import subprocess
    import sys
    import tempfile

    # delete=False keeps the file when the with closes it: only its
    # NAME is needed here, as a path the worker writes and the
    # parent reads back; the finally below removes it
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as tmp:
        result_path = tmp.name
    # dict(os.environ) is a COPY of the environment: the edits below
    # reach only the worker subprocess, never this process
    environment = dict(os.environ)
    environment[_WORKER_FLAG] = "1"
    # OpenMP reads this at library load inside the fresh worker, so
    # the requirement holds for every spawned evaluation
    environment["OMP_NUM_THREADS"] = REQUIRED_OMP_THREADS
    # subprocess.run starts a fresh python (sys.executable = this
    # same interpreter), runs the driver handed to -c, and waits.
    # `"1" if tatt else "0"` encodes each boolean as the string the
    # worker decodes back, `knob or ""` sends the empty string in
    # place of None, and env=environment hands the child the edited
    # COPY of the environment
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER_DRIVER, _THIS_FILE, function,
         example, "1" if tatt else "0", "1" if high_accuracy else "0",
         knob or "", result_path],
        env=environment)
    # the finally below runs on EVERY exit from this try, the raise
    # included, so the temporary result file never outlives the call
    try:
        if completed.returncode != 0:
            raise RuntimeError(
                f"worker for ({function}, {example}, tatt={tatt}) "
                f"exited with code {completed.returncode} before "
                "writing a result; a cosmolike-level abort prints its "
                "reason (e.g. IP::set_mask) just above")
        # json.load parses the worker's file back into the float or
        # two-element list _worker dumped
        with open(result_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


def single_model_chi2(example, tatt, high_accuracy=False, knob=None):
    """chi2 of the frozen fiducial point, evaluated in a fresh worker.

    This is the quantity the reference tests compare against the
    frozen reference and the quantity the generator stores as that
    reference. The evaluation runs in a subprocess (see the module
    docstring for why isolation is mandatory in this project).

    Arguments:
      example = a key of EXAMPLES.
      tatt    = True evaluates the TATT variant, False the NLA one.
      high_accuracy = True evaluates with the pushed numerical
                settings; expect minutes instead of seconds.

    Returns:
      the chi2 as a float.
    """
    # .get returns None when the flag is unset, so only a spawned
    # worker (which carries it as "1") takes the in-process branch
    if os.environ.get(_WORKER_FLAG) == "1":
        return _single_model_chi2_impl(example, tatt,
                                       high_accuracy=high_accuracy,
                                       knob=knob)
    return float(_run_isolated("single", example, tatt,
                               high_accuracy=high_accuracy, knob=knob))


def ten_in_a_row_chi2(example, tatt):
    """Race check, evaluated in one fresh worker subprocess.

    The whole 11-evaluation sequence runs inside ONE worker: the race
    check needs the evaluations to share a model instance, and the
    worker boundary only isolates this configuration from the other
    configurations' dimensions.

    Arguments:
      example = a key of EXAMPLES.
      tatt    = True runs the TATT variant, False the NLA one.

    Returns:
      (fresh, tenth): chi2 of the first fiducial evaluation and chi2
      of the fiducial as the 10th point of the row, both floats.
    """
    # the same worker-or-parent split as single_model_chi2
    if os.environ.get(_WORKER_FLAG) == "1":
        return _ten_in_a_row_impl(example, tatt)
    # the worker's json result is a two-element list; the assignment
    # unpacks it into the two names
    fresh, tenth = _run_isolated("race", example, tatt)
    return float(fresh), float(tenth)
