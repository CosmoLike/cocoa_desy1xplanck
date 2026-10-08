"""Sample the posterior of the hybrid example 1 with Nautilus.

The hybrid examples (the files EXAMPLE_EMUL2_*) use trained emulators in
place of CAMB for the background expansion and the matter power spectra
(use_emulator: 2); cosmolike still computes the survey projections, the
galaxy bias and the intrinsic alignments. This script reads
EXAMPLE_EMUL2_EVALUATE1.yaml (the likelihood desy1xplanck.cosmic_shear
with NLA intrinsic alignments on Y3xPlanckPR4.dataset) and runs Nautilus
nested sampling through cocoa_hybrid_sampling.py: a set of live points
moves from the whole prior toward high likelihood, which estimates the
Bayesian evidence and gives weighted posterior samples. That module, in
external_modules/code/cosmolike_core, documents the method and every
command-line option. The run writes chains/<outroot>.json (with a
converged flag), a checkpoint file and, when posterior samples exist, a
weighted chain; it refuses to overwrite an existing record. Reaching
--maxfeval likelihood calls is not convergence.

From the Cocoa/ folder, with start_cocoa.sh sourced, check the setup
(evaluate the fiducial point, print the order of the sampled parameters,
stop):

    python ./projects/desy1xplanck/EXAMPLE_EMUL2_NAUTILUS1.py --check

then sample with two MPI processes (one coordinates, the other evaluates
the model; the project README shows runs on several nodes):

    mpirun -n 2 --bind-to none python \\
        ./projects/desy1xplanck/EXAMPLE_EMUL2_NAUTILUS1.py \\
        --nlive 1000 --neff 10000 --maxfeval 100000 \\
        --outroot hybrid_nautilus1
"""

from pathlib import Path
import sys

# project = the folder of this file (projects/desy1xplanck). Its parents[1]
# is the Cocoa/ folder, and the / operator of pathlib joins path pieces:
# core is the folder of the shared cosmolike code, put first on the import
# path so that cocoa_hybrid_sampling can be imported.
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))

# Importing cocoa_hybrid_sampling sets the BLAS-thread and CUDA environment
# variables before numpy loads, so it stays the first numerical import.
from cocoa_hybrid_sampling import run


# The block runs only when this file is executed as a script, not when
# another module imports it.
if __name__ == "__main__":
    run(mode="nautilus", project=project, example=1)
