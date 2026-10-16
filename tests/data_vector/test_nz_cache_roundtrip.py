"""n(z) selection cache round-trip checks (shared core suite).

The checks live in cosmolike_core/cocoa_nz_cache_testing.py: live
flips of the z-column reading (Z_LOW left edges, Z_MID sample points),
the reconstruction (midpoint curve, histogram plus Akima), the
preparation factor and the stage-1 interpolant must take effect on the
very next query and restore every value bit for bit when the original
selection returns. This stub only binds those checks to this project's
compiled interface: the machinery under test is the shared core, so
the checks are identical in every project.
"""
import sys
from pathlib import Path

# The compiled interface lives in the project's interface/ folder.
# start_cocoa.sh puts that folder on the import path only for projects
# enabled in set_installation_options.sh, and this check must run for
# any compiled project, so resolve the interface folder and the shared
# core from this file's own location (tests/data_vector sits four
# levels below Cocoa/, which holds external_modules/code).
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "interface"))
_CORE_DIR = str(_HERE.parents[4] / "external_modules" / "code"
                / "cosmolike_core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

import cosmolike_desy1xplanck_interface as ci
from cocoa_nz_cache_testing import build_nz_cache_tests

# pytest collects these module-level names; each callable carries its
# parametrization from the shared builder.
_suite = build_nz_cache_tests(ci)
test_zmid_convention_round_trip = _suite["zmid_convention"]
test_reconstruction_round_trip = _suite["reconstruction"]
test_interpolant_round_trip = _suite["interpolant"]
