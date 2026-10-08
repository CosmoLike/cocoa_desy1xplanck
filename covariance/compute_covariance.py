"""Compute and save the desy1xplanck galaxy/shear covariance from the command line.

A covariance matrix gives the expected scatter of every data-vector
entry and the correlations between entries. This runner computes the
forecast covariance of the galaxy/shear sector of the data vector: the
1500 entries of cosmic shear (xi_+, xi_-), galaxy-galaxy lensing
(gamma_t) and galaxy clustering (w(theta)), each correlation in 30
angular bins, before scale cuts. It does not compute the CMB-lensing
blocks (gk, ks, kk) of the full 1809-entry vector; the likelihood keeps
reading the supplied covariance file. The matrix is saved in four parts:
Gaussian (G), super-sample (SSC), connected non-Gaussian (cNG) and
their total.

The runner reads a cobaya-style evaluate yaml (the fixed cosmology and
the covariance options), calls the C kernels of the compiled interface
through its production bindings, and writes one .npz archive (numpy's
zip file of named arrays). The survey choices (n(z) files, densities,
bias, binning) come from desy1xplanck_covariance.py, the module the
covariance notebook EXAMPLE_EVALUATE_COVARIANCE.ipynb also uses.

From the Cocoa/ folder, with start_cocoa.sh sourced and the project
compiled with covariance generation enabled (see the project README):

    python projects/desy1xplanck/covariance/compute_covariance.py \\
        projects/desy1xplanck/EXAMPLE_EVALUATE_COVARIANCE.yaml

--output PATH writes elsewhere and --overwrite replaces an existing
archive; --help and covariance/README.md describe the measurement-space
and accuracy options. The OpenMP thread count comes from OMP_NUM_THREADS
only.
"""

import os
from pathlib import Path
import sys

# Each BLAS library (OpenBLAS, MKL, the vecLib of macOS) reads its thread
# count from these variables when it loads, so they are set before numpy
# is imported: one BLAS thread per process leaves the cores to cosmolike's
# OpenMP team, whose size comes from OMP_NUM_THREADS in the environment.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies the YAML
# reader; it does not launch MPI workers or a sampler for this calculation.
os.environ["COBAYA_NOMPI"] = "1"

# project = projects/desy1xplanck (parents[1] of this file's path); its
# parents[1] is the Cocoa/ folder, and the / operator of pathlib joins path
# pieces. The shared cosmolike code and this project's interface/ folder go
# first on the import path. Python already put the folder of this script
# there, so desy1xplanck_covariance imports directly.
project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_desy1xplanck_interface as ci
import desy1xplanck_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


# The block runs only when this file is executed as a script. "real" is the
# native measurement space of this project (angular bins); joint=False
# selects the galaxy/shear layout (True is the cluster 6x2pt+N layout of
# another project).
if __name__ == "__main__":
    run_covariance(
        interface=ci, survey=survey, default_space="real", joint=False,
    )
