"""Profile one sampled parameter of the hybrid example 2.

The hybrid examples (the files EXAMPLE_EMUL2_*) use trained emulators in
place of CAMB for the background expansion and the matter power spectra
(use_emulator: 2); cosmolike still computes the survey projections, the
galaxy bias and the intrinsic alignments. This script reads
EXAMPLE_EMUL2_EVALUATE2.yaml (the likelihood desy1xplanck.combo_6x2pt
with NLA intrinsic alignments on Y3xPlanckPR4.dataset). A profile fixes
one sampled parameter at each value of a grid and minimizes -2 log
posterior over all the others, with the annealed search of
cocoa_hybrid_sampling.py; the priors stay in, so the curve is a profile
of the posterior, not of the likelihood alone. The grid is centered on
the minimum saved by EXAMPLE_EMUL2_MINIMIZE2.py (--minfile, required).
cocoa_hybrid_sampling.py, in external_modules/code/cosmolike_core,
documents every command-line option. The run refuses to overwrite an
existing record.

From the Cocoa/ folder, with start_cocoa.sh sourced, check the setup
(evaluate the fiducial point, print the order of the sampled parameters,
stop):

    python ./projects/desy1xplanck/EXAMPLE_EMUL2_PROFILE2.py --check

then profile the first sampled parameter (zero-based index 0) with two
MPI processes (one coordinates, the other evaluates the model):

    mpirun -n 2 --bind-to none python \\
        ./projects/desy1xplanck/EXAMPLE_EMUL2_PROFILE2.py \\
        --profile 0 --nstw 200 --numpts 11 --factor 1 \\
        --minfile ./projects/desy1xplanck/chains/hybrid_min2.json \\
        --outroot hybrid_profile2
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
    run(mode="profile", project=project, example=2)
