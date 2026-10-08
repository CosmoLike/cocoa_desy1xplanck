"""Command line options for these tests (bound from cosmolike_core).

pytest requires a conftest.py inside each project's tests folder (it
discovers the file by walking up from the collected tests), so this
file cannot move; its content is the shared implementation in
cosmolike_core/cocoa_testing.py, bound here the same way
cocoa_test_utils.py binds the test harness. The two options act on
the comparison sweeps only: --high=1 repeats the CFASTPT-vs-FASTPT
comparison at the high-accuracy settings, and --mask selects the
scale-cut mask of the CFASTPT-vs-FASTPT and Halofit-vs-EE2 sweeps (the
choices come from this project's harness, its fastpt_masks tuple:
"frozen" and "ones").
"""

import os
import sys

# The tests folder is not a package; put it on the import path so
# cocoa_test_utils (this project's binding module) resolves no matter
# where pytest was launched from (cocoa_test_utils itself puts
# cosmolike_core on the path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


def pytest_addoption(parser):
    """Register --high and --mask on pytest's parser.

    pytest calls this hook once, before it parses the command line. The
    shared implementation and its documentation live in
    cocoa_testing.conftest_addoption.

    Arguments:
      parser = pytest's option parser (supplied by pytest).

    Returns:
      nothing; the options become readable through config.getoption.
    """
    u._cct.conftest_addoption(parser, u._H.fastpt_masks)


def pytest_configure(config):
    """Copy the option values where the test classes read them.

    pytest calls this hook after it parses the command line. The shared
    implementation (cocoa_testing.conftest_configure) writes the values
    to the environment variables COCOA_FASTPT_HIGH and COCOA_FASTPT_MASK,
    because the tests are unittest classes, which cannot receive pytest
    options directly.

    Arguments:
      config = pytest's configuration object (supplied by pytest).

    Returns:
      nothing; the environment of this process gains the variables.
    """
    u._cct.conftest_configure(config)
