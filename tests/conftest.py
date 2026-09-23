"""Command line options for these tests.

pytest reads every conftest.py on the path down to the tests it
collects, so the option defined here appears next to pytest's own
flags:

    python -m pytest ./projects/desy1xplanck/tests --high=1

The chosen value travels to the test classes through an environment
variable (COCOA_FASTPT_HIGH) rather than a pytest fixture: the tests
are unittest.TestCase classes, and unittest methods cannot receive
pytest fixtures as arguments. An environment variable also lets a
direct `python test_fastpt.py` run set the same switch by exporting
the variable by hand.
"""

import os


def pytest_addoption(parser):
    """Register --high: the accuracy settings of the comparison sweep.

    pytest calls this hook once at startup with its option parser.
    --high=0 (the default) runs the CFASTPT-vs-FASTPT sweep (test
    15) at the frozen default settings; --high=1 repeats it
    with the HIGH_ACCURACY settings of the low-vs-high accuracy
    checks applied to every block. The full comparison is both
    invocations, one without the flag and one with --high=1.

    Arguments:
      parser = pytest's option parser (supplied by pytest).

    Returns:
      nothing; the option becomes readable through config.getoption.
    """
    # action="store" keeps the given text as the option's value;
    # choices rejects anything except the two documented settings
    try:
        parser.addoption(
            "--high", action="store", default="0", choices=("0", "1"),
            help="1 repeats the CFASTPT-vs-FASTPT comparison at the "
                 "HIGH_ACCURACY settings instead of the frozen "
                 "defaults")
    except ValueError:
        # a run collecting several projects' tests at once loads each
        # project's conftest.py, and the second registration of the
        # same option name raises ValueError; the first registration
        # already serves every project, so this one steps aside
        pass


def pytest_configure(config):
    """Copy the --high value where the test classes read it.

    pytest calls this hook after parsing the command line. The value
    lands in the COCOA_FASTPT_HIGH environment variable, which
    test_fastpt.py reads with a "0" default, so a run without the
    option and a run outside pytest behave the same.

    Arguments:
      config = pytest's configuration object (supplied by pytest).

    Returns:
      nothing; the environment of this process gains the variable.
    """
    os.environ["COCOA_FASTPT_HIGH"] = str(config.getoption("--high"))
