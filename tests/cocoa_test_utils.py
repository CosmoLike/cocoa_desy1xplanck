"""Shared harness for the desy1xplanck unit tests: the project's data
bound to the shared Cocoa test machinery.

The machinery itself lives in
external_modules/code/cosmolike_core/cocoa_testing.py: the check of the
stored test state, the chi2 pipeline, the worker subprocesses, the race
and baryon checks, the CFASTPT-vs-FASTPT and Halofit-vs-EE2
comparisons, and the terminal reports. This file holds what belongs to
desy1xplanck alone: the table of examples (cosmic shear and the 6x2pt
and 2x2pt combinations), the TATT test point, the synthetic NLA data
vector every configuration evaluates (NLA_DATASET: the shipped
data_file is real data, far from the minimum of the fiducial point),
the accuracy settings, and the settings and pass limit of the
CFASTPT-vs-FASTPT comparison. It binds them to one
cocoa_testing.CocoaTestHarness object (the harness) and exports the
methods of that object as module-level names, so the test modules and
generate_frozen_reference.py import everything from this module.

Words used in the tests:
  frozen state   tests/frozen/: one snapshot of the configurations, data
                 files and reference chi2 values that the tests evaluate,
                 so that later edits of the project's yaml or data files
                 cannot change what the tests compare.
                 tests/manifest_sha256.json stores the SHA-256
                 fingerprint of every snapshot file, and every test
                 checks the files against it before any model is built;
                 only generate_frozen_reference.py --overwrite rebuilds
                 the snapshot, as a deliberate maintainer action.
  fiducial       the parameter point stored with each configuration.
  worker subprocess
                 a separate Python process that builds one model and
                 hands back its result, so no state of the compiled
                 library carries over from one configuration to the next.
  race check     the fiducial evaluated alone and again after nine other
                 cosmologies on one model: a difference reveals state
                 leaking between evaluations or OpenMP threads racing.
"""

import os
import sys

# ---- tests/ paths -----------------------------------------------------------

# Everything the tests read or write lives relative to this folder, so
# the tests work no matter which directory pytest is launched from
# (__file__ is this module's own path; dirname strips the file name).
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

# ---- the shared machinery ---------------------------------------------------

# The import is path-based (tests/ is three levels below Cocoa/, which
# holds external_modules/code/cosmolike_core) so it works before
# start_cocoa.sh's python-path setup runs.
_CORE_DIR = os.path.abspath(os.path.join(
    TESTS_DIR, "..", "..", "..", "external_modules", "code",
    "cosmolike_core"))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
import cocoa_testing as _cct

# ---- the project data ------------------------------------------------------

# The TATT (Tidal Alignment and Tidal Torquing, an intrinsic-alignment
# model with tidal second-order terms) tests replace these values in
# the frozen point. In the NLA reference point the amplitudes DES_A2_1
# and DES_BTA_1 are zero (DES_A2_2, the redshift index of A2, keeps its
# value), so the nonzero amplitudes here make the TATT reference
# exercise the second-order terms.
TATT_POINT = {
    "DES_A2_1": 0.05,
    "DES_BTA_1": 0.05,
    "DES_A2_2": -1.51541,
}

# The TATT variants evaluate against a data vector generated with TATT
# at the fiducial point. Reason: against an NLA-based vector the TATT
# chi2 sits away from its minimum, where it responds linearly (not
# quadratically) to tiny numerical changes, so harmless rounding-level
# shifts would use up much of the 0.2 chi2 band the reference tests
# allow. All examples share one data set, so a single full-length
# vector generated from the example2 TATT model serves every
# configuration (the mask of each probe selects its section).
# TATT_GENERATORS records that choice as {descriptor file: source
# example}; no code reads it (the generator reads SYNTHETIC_VECTORS).
TATT_GENERATORS = {
    "tatt_desy1xplanck.dataset": "example2",
}

# This project's shipped data_file is real data, and the example
# cosmology is not its best fit: the chi2 sits far from the minimum
# (hundreds to thousands), where it responds linearly to tiny theory
# changes. A chi2 comparison evaluated there reports large shifts that
# say nothing about the numerics near a fit. The NLA variants therefore
# evaluate against a synthetic data vector, generated with the default
# (NLA) model at the fiducial point of example2 when the frozen state is
# built, like the TATT vector: at its own minimum the chi2 responds
# quadratically and stays stable.
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
# settings of the likelihood pushed far beyond the defaults (the CAMB
# side is HIGH_ACCURACY_CAMB_EXTRA_ARGS of cocoa_testing).
HIGH_ACCURACY_LIKELIHOOD = {
    # the all-settings check compares the default against boost 3; the
    # one-at-a-time scan (ACCURACY_KNOBS below) also runs boost 5 as a
    # stress value
    "accuracyboost": 3.0,       # default 1.0
    "internal_accuracyboost": 2.0, # default 1.0 (denser convolution grid)
    "integration_accuracy": 10,  # default 0
    "lmax": 200000,             # default 50000-75000
    "kmax_boltzmann": 40.0,     # default 5.0-7.5
}

# The one-at-a-time scan of test_accuracy.py: each entry is (label,
# likelihood overrides, camb extra_args overrides), evaluated alone on
# the example2 NLA configuration before the all-settings checks, so a
# large all-settings delta can be traced to the setting causing it. The
# accuracyboost 5 entry is a stress value beyond what measuring the
# default numerics needs: it would expose a cosmolike table whose size
# does not follow the boost.
# Investigation order when several settings move the chi2: raise the
# cosmolike accuracyboost first (cheap), then camb k_per_logint, and
# only then camb AccuracyBoost (expensive at run time): an apparent
# CAMB sensitivity can masquerade as unresolved cosmolike-side
# resolution, so the cheap settings must be settled before the
# expensive one is blamed. kmax_boltzmann and camb kmax are one physical
# cutoff seen from the two sides, so the scan moves them together.
ACCURACY_KNOBS = [
    ("accuracyboost -> 3", {"accuracyboost": 3.0}, {}),
    ("accuracyboost -> 5 (stress)", {"accuracyboost": 5.0}, {}),
    ("internal_accuracyboost 1->2", {"internal_accuracyboost": 2.0}, {}),
    ("integration_accuracy -> 10", {"integration_accuracy": 10}, {}),
    ("lmax -> 200000", {"lmax": 200000}, {}),
    ("kmax_boltzmann -> 40 + camb kmax -> 50",
     {"kmax_boltzmann": 40.0}, {"kmax": 50.0}),
    ("camb AccuracyBoost -> 2", {}, {"AccuracyBoost": 2.0}),
    ("camb k_per_logint -> 50", {}, {"k_per_logint": 50}),
]

# The three stored configurations: cosmic shear (example1), 6x2pt
# (example2) and 2x2pt (example2_2x2pt: example2's configuration with
# the likelihood renamed). "frozen_module" is the configuration's file
# in frozen/; "provenance" names the example yaml it was made from (kept
# in frozen/ for a human to compare, never loaded); "likelihood" is the
# cobaya component name, needed to reach that block inside the loaded
# info dictionary; "source_likelihood" is the name in the provenance
# yaml when it differs; "tatt_dataset" is the data set of the TATT
# variants.
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

# Pass limit on the covariance-weighted difference of the two
# implementations: at each point both blocks print their theory data
# vector, and the tested number is delta^T C^-1 delta with
# delta = dv(FASTPT low) - dv(CFASTPT) and C^-1 the masked inverse
# covariance: the chi2 of the implementation difference, zero when the
# vectors agree. The raw chi2 values are printed only as information:
# across the IA prior they are large, so their difference follows the
# local chi2 slope and measures the distance from the data, not the
# numerics. 0.2 is the band of the chi2 reference tests
# (CHI2_TOLERANCE), and FASTPT_LOW_SETTINGS meets it with the converged
# two-grid configuration: this project's sweep measures at most
# Delta chi2 = 0.000239 there, against Delta chi2 = 29.6 across the
# prior when one grid serves both the output table and the
# convolutions.
FASTPT_COMPARISON_TOLERANCE = 0.2

# The python FAST-PT side has numerical settings of its own, read by
# the fastpt theory block from its extra_args block
# (external_modules/code/PyFAST-PT/fastpt.py, symlinked into cobaya
# as theories/fastpt). The block computes on two grids: accuracyboost
# multiplies the density of the output table cosmolike reads with
# linear interpolation (the accuracy driver), and
# internal_accuracyboost the density of the internal grid the FFTLog
# convolutions run on; a cubic spline in log k upsamples the terms
# from one grid onto the other. Both boosts default to 1.0 = the
# converged configuration, so the low settings are the defaults; they
# are written out here so the test keeps evaluating this exact
# configuration even if the defaults move. High doubles both boosts, so
# the advisory column shows the residual grid response of low.
FASTPT_LOW_SETTINGS = {
    "accuracyboost": 1.0,
    "internal_accuracyboost": 1.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

FASTPT_HIGH_SETTINGS = {
    "accuracyboost": 2.0,
    "internal_accuracyboost": 2.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

# ---- project-independent constants ------------------------------------------

# These are identical in every project and live in the core module.
REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
RACE_TOLERANCE = _cct.RACE_TOLERANCE
RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
HIGH_ACCURACY_CAMB_EXTRA_ARGS = _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS
BARYON_METHODS = _cct.BARYON_METHODS
BARYON_POINT_OVERRIDES = _cct.BARYON_POINT_OVERRIDES
NONLINEAR_COMPARISON_POINTS = _cct.NONLINEAR_COMPARISON_POINTS

# The 30 CFASTPT-vs-FASTPT comparison points under this project's
# sampled-parameter prefix; the values are identical in every project.
FASTPT_COMPARISON_POINTS = _cct.fastpt_comparison_points("DES")

# ---- the harness -----------------------------------------------------------

# One instance binds the shared machinery to this project's data. The
# assignments below give the module functions of cocoa_testing and the
# methods of this instance module-level names (a bound method remembers
# its instance, so u.verify_frozen() runs _H.verify_frozen()): the names
# the test modules and generate_frozen_reference.py import.
_H = _cct.CocoaTestHarness(
    worker_file=__file__,
    interface_module="cosmolike_desy1xplanck_interface",
    examples=EXAMPLES,
    tatt_point=TATT_POINT,
    accuracy_knobs=ACCURACY_KNOBS,
    high_accuracy_likelihood=HIGH_ACCURACY_LIKELIHOOD,
    fastpt_low_settings=FASTPT_LOW_SETTINGS,
    fastpt_high_settings=FASTPT_HIGH_SETTINGS,
    fastpt_points=FASTPT_COMPARISON_POINTS,
    nla_dataset=NLA_DATASET,
    fastpt_masks=("frozen", "ones"),
)

# ---- module functions re-exported from the core (no project state) ----------
require_cocoa_environment = _cct.require_cocoa_environment
assert_omp_threads = _cct.assert_omp_threads
sha256_of = _cct.sha256_of
make_model = _cct.make_model
evaluate_chi2 = _cct.evaluate_chi2
_evaluate_cached = _cct._evaluate_cached
_load_datavector = _cct._load_datavector
_baryon_method = _cct._baryon_method
_baryon_dataset = _cct._baryon_dataset
report_chi2_test = _cct.report_chi2_test
report_race_test = _cct.report_race_test
report_accuracy = _cct.report_accuracy
report_knob = _cct.report_knob
report_fastpt_comparison = _cct.report_fastpt_comparison
report_nonlinear_comparison = _cct.report_nonlinear_comparison

# ---- bound methods of the harness (the machinery, project-bound) ------------
compute_manifest = _H.compute_manifest
verify_frozen = _H.verify_frozen
load_reference = _H.load_reference
_frozen_module = _H._frozen_module
load_frozen_info = _H.load_frozen_info
load_frozen_point = _H.load_frozen_point
build_point = _H.build_point
_single_model_chi2_impl = _H._single_model_chi2_impl
_ten_in_a_row_impl = _H._ten_in_a_row_impl
_baryon_accuracy_delta_impl = _H._baryon_accuracy_delta_impl
_baryon_drift_chi2_impl = _H._baryon_drift_chi2_impl
single_model_chi2 = _H.single_model_chi2
ten_in_a_row_chi2 = _H.ten_in_a_row_chi2
baryon_accuracy_delta = _H.baryon_accuracy_delta
baryon_drift_chi2 = _H.baryon_drift_chi2
_worker = _H._worker
_run_isolated = _H._run_isolated
_fastpt_comparison_info = _H._fastpt_comparison_info
_fastpt_comparison_block = _H._fastpt_comparison_block
_run_fastpt_comparison_worker = _H._run_fastpt_comparison_worker
cfastpt_vs_fastpt_chi2s = _H.cfastpt_vs_fastpt_chi2s
_nonlinear_comparison_block = _H._nonlinear_comparison_block
halofit_vs_ee2_dchi2s = _H.halofit_vs_ee2_dchi2s
