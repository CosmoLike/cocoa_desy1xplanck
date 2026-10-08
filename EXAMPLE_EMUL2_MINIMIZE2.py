"""desy1xplanck hybrid minimize; see the project README for MPI examples."""

from pathlib import Path
import sys

project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


if __name__ == "__main__":
    run(mode="minimize", project=project, example=2)
