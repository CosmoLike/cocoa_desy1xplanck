"""Command line options for these tests (bound from cosmolike_core).

pytest requires a conftest.py inside each project's tests folder (it
discovers the file by walking up from the collected tests), so this
file cannot move; its content is the shared implementation in
cosmolike_core/cocoa_testing.py, bound here the same way
cocoa_test_utils.py binds the test harness. The --mask choices come
from this project's harness (its fastpt_masks tuple).
"""

import os
import sys

# The tests folder is not a package; put it on the import path so the
# project shim resolves no matter where pytest was launched from (the
# shim itself puts cosmolike_core on the path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


def pytest_addoption(parser):
    """Register --high and --mask on pytest's parser.

    The shared implementation and its documentation live in
    cocoa_testing.conftest_addoption.

    Arguments:
      parser = pytest's option parser (supplied by pytest).

    Returns:
      nothing; the options become readable through config.getoption.
    """
    u._cct.conftest_addoption(parser, u._H.fastpt_masks)


def pytest_configure(config):
    """Copy the option values where the test classes read them.

    The shared implementation and its documentation live in
    cocoa_testing.conftest_configure.

    Arguments:
      config = pytest's configuration object (supplied by pytest).

    Returns:
      nothing; the environment of this process gains the variables.
    """
    u._cct.conftest_configure(config)
