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
    files at freeze time, kept only so a human can diff how the live
    examples drifted; no test reads them.

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
EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "desy1xplanck.cosmic_shear",
    },
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "desy1xplanck.combo_6x2pt",
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
    with open(path, "rb") as f:
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
    for base, dirs, names in os.walk(FROZEN_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(names):
            if name == ".DS_Store" or name.endswith(".pyc"):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, TESTS_DIR).replace(os.sep, "/")
            files[rel] = sha256_of(full)
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
    with open(MANIFEST_FILE) as f:
        expected = json.load(f)["files"]
    actual = compute_manifest()
    problems = []
    for rel, digest in expected.items():
        if rel not in actual:
            problems.append(f"MISSING  {rel}")
        elif actual[rel] != digest:
            problems.append(f"CHANGED  {rel}")
    for rel in actual:
        if rel not in expected:
            problems.append(f"EXTRA    {rel}")
    if problems:
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
    spec = importlib.util.spec_from_file_location(f"frozen_{example}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_info(example, tatt):
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
      example = "example1" or "example2" (a key of EXAMPLES).
      tatt    = True selects the TATT IA model, False keeps NLA.

    Returns:
      the input dictionary ready for cobaya's get_model.
    """
    from cobaya.yaml import yaml_load

    cfg = EXAMPLES[example]
    info = yaml_load(_frozen_module(example).yaml_string)
    info.pop("sampler", None)
    info.pop("output", None)
    info["debug"] = 30
    info["timing"] = False
    likelihood_block = info["likelihood"][cfg["likelihood"]]
    likelihood_block["path"] = os.path.join(FROZEN_DIR, "data")
    likelihood_block["IA_model"] = 1 if tatt else 0
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
    return dict(_frozen_module(example).point)


def build_point(model, example, tatt):
    """Assemble the exact point a test evaluates, with a drift check.

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
    point = load_frozen_point(example)
    sampled = set(model.parameterization.sampled_params())
    if sampled != set(point):
        raise AssertionError(
            "sampled-parameter set differs from the frozen point (the "
            "likelihood/theory code changed its parameters):\n"
            f"  new since freeze: {sorted(sampled - set(point))}\n"
            f"  gone since freeze: {sorted(set(point) - sampled)}"
        )
    if tatt:
        for name, value in TATT_POINT.items():
            if name not in point:
                raise ValueError(f"TATT parameter {name} is not sampled")
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

    posterior = model.logposterior(point, cached=False)
    if len(posterior.loglikes) != 1:
        raise RuntimeError(
            f"expected exactly one likelihood: {posterior.loglikes}")
    chi2 = -2.0 * posterior.loglikes[0]
    if not np.isfinite(chi2):
        raise AssertionError(f"non-finite chi2 at point {point}")
    return float(chi2)


def single_model_chi2(example, tatt):
    """chi2 of the frozen fiducial point on a freshly built model.

    This is the quantity tests 1, 3, 5, and 7 compare against the
    frozen reference, and the quantity the generator stores as that
    reference.

    Arguments:
      example = "example1" or "example2" (a key of EXAMPLES).
      tatt    = True evaluates the TATT variant, False the NLA one.

    Returns:
      the chi2 as a float.
    """
    ia_label = "TATT" if tatt else "NLA"
    print(f"  building model ({example}, {ia_label}) ...", flush=True)
    info = load_frozen_info(example, tatt)
    model = make_model(info)
    point = build_point(model, example, tatt)
    print("  evaluating the fiducial point ...", flush=True)
    return evaluate_chi2(model, point)


def ten_in_a_row_chi2(example, tatt):
    """Race check: the fiducial evaluated fresh and as 10th of a row.

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
    ia_label = "TATT" if tatt else "NLA"
    print(f"  building model ({example}, {ia_label}) ...", flush=True)
    info = load_frozen_info(example, tatt)
    model = make_model(info)
    point = build_point(model, example, tatt)
    fresh = evaluate_chi2(model, point)
    print(f"  fresh model, fiducial point:  chi2 = {fresh:.8f}", flush=True)
    for i, perturbation in enumerate(RACE_PERTURBATIONS, start=1):
        chi2 = evaluate_chi2(model, {**point, **perturbation})
        changed = ", ".join(f"{k}={v}" for k, v in perturbation.items())
        print(f"  row {i:2d}/10 ({changed}):  chi2 = {chi2:.4f}", flush=True)
    tenth = evaluate_chi2(model, point)
    print(f"  row 10/10 (fiducial again):  chi2 = {tenth:.8f}", flush=True)
    return fresh, tenth
