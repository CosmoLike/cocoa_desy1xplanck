"""Command line options for these tests.

pytest reads every conftest.py on the path down to the tests it
collects, so the options defined here appear next to pytest's own
flags:

    python -m pytest ./projects/desy1xplanck/tests --high=1 --mask=ones

The chosen values travel to the test classes through environment
variables (COCOA_FASTPT_HIGH, COCOA_FASTPT_MASK) rather than pytest
fixtures: the tests are unittest.TestCase classes, and unittest
methods cannot receive pytest fixtures as arguments. An environment
variable also lets a direct `python test_fastpt.py` run set the same
switch by exporting the variable by hand.
"""

import os

# the masks the comparison sweep can run under; "frozen" keeps the
# shipped 6x2pt scale-cut mask of the frozen contract, and "ones"
# keeps every data point (no scale cuts). The mapping to dataset
# descriptors lives in the shared harness (fastpt_masks in
# cocoa_test_utils).
MASK_CHOICES = ("frozen", "ones")


def pytest_addoption(parser):
    """Register --high and --mask: the comparison sweep's switches.

    pytest calls this hook once at startup with its option parser.
    --high=0 (the default) runs the CFASTPT-vs-FASTPT sweeps (tests
    15-17) at the frozen default settings; --high=1 repeats them with
    the HIGH_ACCURACY settings of the low-vs-high accuracy checks
    applied to every block. The full comparison is both invocations,
    one without the flag and one with --high=1.

    --mask selects the scale-cut mask of the same sweeps:
    --mask=frozen (the default) keeps the shipped 6x2pt mask of the
    frozen contract, and --mask=ones keeps every data point (no
    scale cuts), the strictest comparison.

    Arguments:
      parser = pytest's option parser (supplied by pytest).

    Returns:
      nothing; the options become readable through config.getoption.
    """
    # action="store" keeps the given text as the option's value;
    # choices rejects anything except the documented settings. A run
    # collecting several projects' tests at once loads each project's
    # conftest.py, and the second registration of the same option
    # name raises ValueError; the first registration already serves
    # every project, so the except clause steps aside.
    try:
        parser.addoption(
            "--high", action="store", default="0", choices=("0", "1"),
            help="1 repeats the CFASTPT-vs-FASTPT comparison at the "
                 "HIGH_ACCURACY settings instead of the frozen "
                 "defaults")
    except ValueError:
        pass
    try:
        parser.addoption(
            "--mask", action="store", default="frozen",
            choices=MASK_CHOICES,
            help="the scale-cut mask of the CFASTPT-vs-FASTPT "
                 "comparison: frozen (the shipped 6x2pt contract "
                 "mask, the default) or ones (every data point kept)")
    except ValueError:
        pass


def pytest_configure(config):
    """Copy the option values where the test classes read them.

    pytest calls this hook after parsing the command line. The values
    land in the COCOA_FASTPT_HIGH and COCOA_FASTPT_MASK environment
    variables (see the module docstring for why environment
    variables), which test_fastpt.py reads with the "0" and "frozen"
    defaults, so a run without the options and a run outside pytest
    behave the same.

    Arguments:
      config = pytest's configuration object (supplied by pytest).

    Returns:
      nothing; the environment of this process gains the variables.
    """
    os.environ["COCOA_FASTPT_HIGH"] = str(config.getoption("--high"))
    os.environ["COCOA_FASTPT_MASK"] = str(config.getoption("--mask"))
