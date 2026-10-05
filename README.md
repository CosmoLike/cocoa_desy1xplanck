# Table of contents <a name="table_of_contents"></a>

1. [Running Cosmolike projects (Basic instructions)](#desy1xplanck_running_cosmolike_projects)
2. [Baryonic feedback on EXAMPLE_EVALUATE1](#desy1xplanck_baryonic_feedback)
3. [Running Hybrid Cosmolike-ML emulators](#desy1xplanck_examples_emul2)
4. [Unit tests](#desy1xplanck_unit_tests)
5. [Minimum accuracy parameters](#desy1xplanck_minimum_accuracy)
6. [Computing covariances](#computing_covariances)

## Running Cosmolike projects (Basic instructions) <a name="desy1xplanck_running_cosmolike_projects"></a> 

> [!WARNING]
> **CLI for production; notebook wrappers for exploration.**
>
> Run production and HPC calculations from YAML through the optimized
> `_interface` bindings. Notebook `_wrapper` APIs expose intermediate
> quantities for exploration; copying and rearranging their arrays adds
> overhead. Both routes call the same C kernels.
>
> In a matched **LSST Y1 covariance** test on an M2 Pro with eight threads,
> the CLI averaged **68.34 s** (three runs); one wrapper run took **177.74 s**.
> The CLI was **2.60× faster**, with bitwise-identical covariance components.
> See [the production covariance CLI](#computing_covariances).

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment the following lines on `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which cosmolike projects will be installed and compiled
>     # ------------------------------------------------------------------------------
>     #export IGNORE_COSMOLIKE_LSST_Y1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     (...)
>     export IGNORE_COSMOLIKE_DESXPLANCK_CODE=1
>
>     (...)
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export DESXPLANCK_URL="https://git@github.com/CosmoLike/cocoa_desy1xplanck.git"
>     export DESXPLANCK_GIT_NAME="desy1xplanck"
>     #Pin the project version with at most one of the keys below (COMMIT, BRANCH, or TAG).
>     #If more than one is set, COMMIT wins over BRANCH, and BRANCH wins over TAG.
>     #If none is set, Cocoa loads the latest commit on the repository default branch.
>     #export DESXPLANCK_GIT_BRANCH="main"
>     #export DESXPLANCK_GIT_COMMIT="abc"
>     export DESXPLANCK_GIT_TAG="v4.10.4"

> [!NOTE]
> In case users need to rerun `setup_cocoa.sh`, Cocoa will not download previously installed packages, cosmolike projects, or large datasets, unless the following keys are set on `set_installation_options.sh`
>
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     # ------------------------------------------------------------------------------
>     # OVERWRITE_EXISTING_XXX_CODE=1 -> setup_cocoa overwrites existing PACKAGES ----
>     # overwrite: delete the existing PACKAGE folder and install it again -----------
>     # redownload: delete the compressed file and download data again ---------------
>     # These keys are only relevant if you run setup_cocoa multiple times -----------
>     # ------------------------------------------------------------------------------
>     (...)
>     export OVERWRITE_EXISTING_ALL_PACKAGES=1    # except cosmolike projects
>     #export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 # dangerous (possible loss of uncommitted work)
>                                                 # if unset, users must manually delete cosmolike projects
>     #export REDOWNLOAD_EXISTING_ALL_DATA=1      # warning: some data is many GB

> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
> 
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
> 
>       source ./installation_scripts/compile_all_projects.sh
> 
> or (in case users just want to compile desy1xplanck project)
>
>       source ./projects/desy1xplanck/scripts/compile_desy1xplanck.sh

> [!TIP]
> Assuming Cocoa is installed on a local (not remote!) machine, type the command below after step 2️⃣ to run Jupyter Notebooks.
>
>     jupyter notebook --no-browser --port=8888
>
> The terminal will then show a message similar to the following template:
>
>     (...)
>     [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
>     [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>     [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
>
> Now go to the local internet browser and type `http://127.0.0.1:8888/?token=XXX`, where XXX is the previously saved token displayed on the line
> 
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>
> The project desy1xplanck contains jupyter notebook examples located at `projects/desy1xplanck`.

To run the example

**Step :one:**: activate the Cocoa Conda environment,  and the private Python environment 

      conda activate cocoa

and

      source start_cocoa.sh
 
**Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

**Step :three:**: The folder `projects/desy1xplanck` contains examples. So, run the `cobaya-run` on the first example following the commands below.

> [!Warning] 
> (Linux only) In some HPC nodes, `numa` can cause you problems. If that is the case,
> replace `numa` with `slot`

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_EVALUATE1.yaml -f

  - macOS (arm)

        mpirun -n 1 --oversubscribe \
         cobaya-run ./projects/desy1xplanck/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_MCMC1.yaml -f

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_MCMC1.yaml -f

The likelihoods of the examples, the parameter files they include, and the parameters each likelihood fixes or must not vary are described in [likelihood/README.md](likelihood/README.md).


> [!Warning]
> CosmoLike supports the optimized strict-IEEE default build and
> `COSMOLIKE_DEBUG_MODE`. The compiler mode `COSMOLIKE_AGGRESSIVE_MODE`
> is retired because its fast-math configuration produced incorrect
> covariance inverses. Unset that variable before compiling.
> Do not enable `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`,
> `-fassociative-math`, `-ffinite-math-only`, `-freciprocal-math`,
> `-fno-signed-zeros`, or `-fno-trapping-math` in CosmoLike builds.
> This does not change Cocoa's separate `--aggressive` download option.


# Baryonic feedback on EXAMPLE_EVALUATE1 <a name="desy1xplanck_baryonic_feedback"></a>

`EXAMPLE_EVALUATE1.yaml` can apply an external baryonic feedback suppression to the
matter power spectrum via the `bfmt` theory block (SP(k), BCEmu, Flamingo, BACCOemu,
or BCemu2025). By default, the example runs without feedback.

**Step :one:**: ensure the lines below are commented out in `set_installation_options.sh`
before running `setup_cocoa.sh` and `compile_cocoa.sh`. *By default, these lines should
be commented out, but it is worth checking*.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      #export IGNORE_PYSPK_CODE=1     # SP(k)
      #export IGNORE_BCEMU_CODE=1     # BCEmu
      #export IGNORE_FBRE_CODE=1      # FlamingoBaryonResponseEmulator
      #export IGNORE_BACCOEMU_CODE=1  # BACCOemu
      #export IGNORE_BFMT_CODE=1      # Baryon Feedback Theory Block

**Step :two:**: in `EXAMPLE_EVALUATE1.yaml`, uncomment the `bfmt` theory block and select
the model:

      theory:
        bfmt:
          baryon_model: 2 # 1 = SP(k), 2 = BCEmu, 3 = FlamingoEmulator, 4 = BACCOemu, 5 = BCemu2025

**Step :three:**: set `external_baryon_suppression: True` on the `desy1xplanck.cosmic_shear`
likelihood block.

**Step :four:**: uncomment the selected model's parameters in the `params` block and in
the `sampler: evaluate: override` block (the example carries a commented block for each
model).

> [!TIP]
> For the sampled parameters of each model, their validity ranges, and the `bfmt`
> options, see `Cocoa/external_modules/code/baryon_suppression/README.md`.

# Running Hybrid Cosmolike-ML emulators <a name="desy1xplanck_examples_emul2"></a>

> [!Warning]
> The code and examples associated with this section are still in alpha stage

Our main line of research involves emulators that simulate the entire Cosmolike data vectors, 
and each project (LSST, Roman, DES) contains its own README with emulator examples. 
The speed of such emulators is incredible, especially when GPUs are available, 
and our emulators do take advantage of the CPU-GPU integration on Apple MX chips.

While the data vector emulators are incredibly fast, there is an intermediate 
approach that emulates only the Boltzmann outputs (comoving distance, linear and 
nonlinear matter power spectrum). This hybrid-ML case can offer greater flexibility, 
especially in the initial phases of a research project, as changes to the modeling 
of nuisance parameters or to the assumed galaxy distributions do not require 
retraining of the network. 

Examples in the hybrid case all have the prefix **EXAMPLE_EMUL2** (note the `2`). The required flags on `set_installation_options.sh` are similar to what we showed in the previous emulator section.

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores. Below, we set it to 4, the ideal setting for hybrid examples.

  - Linux

        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:**: Remove GPU (idea is to run emulators on the CPU!)

  - Linux

        export CUDA_VISIBLE_DEVICES=""

 **Step :four:** Run `cobaya-run` on the first emulator example, following the commands below.

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_EMUL2_EVALUATE1.yaml -f

  - macOS (arm)
    
        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_EMUL2_EVALUATE1.yaml -f
    
- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_EMUL2_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_EMUL2_MCMC1.yaml -r

> [!NOTE]
> **Running on more than one node.** The flag `--mca btl vader,tcp,self` works unchanged across
> nodes: Open MPI picks the transport per pair of ranks, using shared memory (`vader`) within a
> node and TCP between nodes. Three things deserve attention on multi-node runs:
>
> 1. **Network interface.** The TCP layer must not select an interface that is not routable
>    between compute nodes. The flag `--mca btl_tcp_if_exclude lo,docker0,virbr0,ib0` excludes
>    the common offenders. TCP bandwidth is not a limitation for our workloads, which exchange
>    small, infrequent MPI messages.
>
> 2. **Environment.** Ranks on remote nodes must see Cocoa's environment (`ROOTDIR`, `PATH`,
>    `LD_LIBRARY_PATH`, `PYTHONPATH`, `CONDA_PREFIX`, the OpenMP/BLAS thread settings, and
>    `CLIK_PATH`/`CLIK_DATA`/`CLIK_PLUGIN`). Slurm forwards the submitting environment
>    automatically; the explicit `-x` flags in our sbatch templates repeat this so the
>    scripts also work under ssh-based launchers. No other Cocoa installation flags are read at runtime.
>
> 3. **Slurm geometry.** Keep `ntasks-per-node` × `cpus-per-task` no larger than the cores per
>    node, and use `--map-by numa:pe=${OMP_NUM_THREADS}` so each rank reserves the cores its
>    OpenMP threads will use.

> [!NOTE]
> **Note on core oversubscription**: an MPI process that is waiting still burns 100% of its
> core, checking for messages in a loop. With more processes than cores, this stalls the
> processes doing real work. Open MPI usually detects this and makes waiting processes give
> up the CPU, but its detection can be fooled. Adding `--mca mpi_yield_when_idle 1` forces
> that behavior; it is harmless otherwise.

The `Nautilus`, `Minimizer`, and `Profile` scripts below contain an internally 
defined `yaml_string` that specifies priors, 
likelihoods, and the theory code, all following Cobaya Conventions. 
The `PolyChord` example, in contrast, is configured directly by the YAML file `EXAMPLE_EMUL2_POLY1.yaml`.

- **Nautilus**:

  - Linux 
    
        export OMP_NUM_THREADS=1

        "${CONDA_PREFIX}"/bin/mpirun -n 96 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
          -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
          -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
          -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          python -m mpi4py.futures ./projects/desy1xplanck/EXAMPLE_EMUL2_NAUTILUS1.py \
            --root ./projects/desy1xplanck/ --outroot "EXAMPLE_EMUL2_NAUTILUS1"  \
            --maxfeval 750000 --nlive 2048 --neff 15000 \
            --flive 0.01 --nnetworks 5

  - macOS (arm)

        export OMP_NUM_THREADS=1

        mpirun -n 12 --oversubscribe \
          python -m mpi4py.futures ./projects/desy1xplanck/EXAMPLE_EMUL2_NAUTILUS1.py \
            --root ./projects/desy1xplanck/ \
            --outroot "EXAMPLE_EMUL2_NAUTILUS1" \
            --maxfeval 750000 --nlive 2048 --neff 15000 \
            --flive 0.01 --nnetworks 5

- **PolyChord**:

  - Linux (assuming node with 96 cores)

        export OMP_NUM_THREADS=4

        "${CONDA_PREFIX}"/bin/mpirun -n 24 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
          -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
          -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
          -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_EMUL2_POLY1.yaml -r

  - macOS (arm)

        export OMP_NUM_THREADS=1

        mpirun -n 12 --oversubscribe \
          cobaya-run ./projects/desy1xplanck/EXAMPLE_EMUL2_POLY1.yaml -r

- **Global Minimizer**:

  Our minimizer is a reimplementation of `Procoli`, developed by Karwal et al (arXiv:2401.14225) 

  - Linux (assuming node with 96 cores)

        export OMP_NUM_THREADS=4

        "${CONDA_PREFIX}"/bin/mpirun -n 24 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
          -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
          -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
          -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          python ./projects/desy1xplanck/EXAMPLE_EMUL2_MINIMIZE1.py \
            --root ./projects/desy1xplanck/ \
            --outroot "EXAMPLE_EMUL2_MIN1" \
            --nstw 350

  - macOS (arm)

        export OMP_NUM_THREADS=1

        mpirun -n 12 --oversubscribe \
          python ./projects/desy1xplanck/EXAMPLE_EMUL2_MINIMIZE1.py \
            --root ./projects/desy1xplanck/ \
            --outroot "EXAMPLE_EMUL2_MIN1" \
            --nstw 350
    
    
  The number of steps per Emcee walker per temperature is $n_{\rm stw}$,
  and the number of walkers is $n_{\rm w}={\rm max}(3n_{\rm params},n_{\rm MPI})$.
  The minimum number of total evaluations is $3n_{\rm params} \times n_{\rm T} \times n_{\rm stw}$, which can be distributed among $n_{\rm MPI} = 3n_{\rm params}$ MPI processes for faster results.

- **Profile**: 

  - Linux (assuming node with 96 cores)

        export OMP_NUM_THREADS=4

        "${CONDA_PREFIX}"/bin/mpirun -n 24 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
          -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
          -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
          -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          python ./projects/desy1xplanck/EXAMPLE_EMUL2_PROFILE1.py \
            --root ./projects/desy1xplanck/ --cov 'chains/EXAMPLE_EMUL2_MCMC1.covmat' \
            --outroot "EXAMPLE_EMUL2_PROFILE1" \
            --factor 3 --nstw 350 --numpts 10 \
            --profile 1 \
            --minfile="./projects/desy1xplanck/chains/EXAMPLE_EMUL2_MIN1.txt"

  - macOS (arm)
        
        export OMP_NUM_THREADS=1
        
        mpirun -n 12 --oversubscribe \
          python ./projects/desy1xplanck/EXAMPLE_EMUL2_PROFILE1.py \
            --root ./projects/desy1xplanck/ \
            --cov 'chains/EXAMPLE_EMUL2_MCMC1.covmat' \
            --outroot "EXAMPLE_EMUL2_PROFILE1" \
            --factor 3 --nstw 350 --numpts 10 --profile 1 \
            --minfile="./projects/desy1xplanck/chains/EXAMPLE_EMUL2_MIN1.txt"

Details on the matter power spectrum emulator designs will be presented in the 
[emulator_code](https://github.com/SBU-COSMOLIKE/emulators_code) repository. 

Basically, we apply standard neural network techniques to generalize 
the *syren-new* Eq. 6 of [arXiv:2410.14623](https://arxiv.org/abs/2410.14623) 
formula for the linear power spectrum (w0waCDM with a fixed neutrino mass of $0.06$ eV) 
to new models, extended ranges, or higher precision. 
Similarly, we use networks to generalize the *syren-Halofit* LCDM nonlinear 
boost fit (Eq. 11 of [arXiv:2402.17492](https://arxiv.org/abs/2402.17492)).

# Unit tests <a name="desy1xplanck_unit_tests"></a>

The `tests/` folder holds unit tests for the likelihoods of this
project: they compare each likelihood against stored reference
values, check for race conditions from OpenMP threading, and measure
the numerical error of the default accuracy settings. The
tests read nothing from the live project;
[tests/README.md](tests/README.md) describes every test, the tests'
own data snapshot, and how to refresh it.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the tests of this project

    python -m pytest ./projects/desy1xplanck/tests/data_vector

## Minimum accuracy parameters <a name="desy1xplanck_minimum_accuracy"></a>

The advisory checks in `tests/data_vector/test_accuracy.py` measure the
numerical error of the default accuracy settings: each setting is
raised one at a time on the 6x2pt configuration, so a large
$\Delta\chi^2$ can be attributed to the setting causing it, and
then every setting at once.

Each check prints the $\Delta\chi^2$ between the high-accuracy and
the default evaluations, to compare against the 0.2 band the
reference tests allow. No measured values are quoted here: rerun
the checks to measure them on the current code, and see
[tests/README.md](tests/README.md) for each check, the settings
raised, and what each setting controls.

The `accuracyboost <= 3` warnings next to the `accuracyboost` lines
in this project's yaml files describe cosmolike builds whose FFTLog
zero-padding stays constant while the boost densifies the chi grid;
the cosmolike core compiled here scales the padding with the grid
(`external_modules/code/cosmolike/cosmo2D.c`).

# Computing covariances <a name="computing_covariances"></a>

[EXAMPLE_EVALUATE_COVARIANCE.ipynb](EXAMPLE_EVALUATE_COVARIANCE.ipynb)
computes a covariance with this project's 3×2pt measurement layout.
It keeps G, SSC and cNG separately, applies the supplied likelihood mask,
and plots the computed and supplied totals together.

| Measurement choice | Notebook example |
| --- | --- |
| Dataset | [data/Y3xPlanckPR4.dataset](data/Y3xPlanckPR4.dataset) |
| Primary space | Real-space 3×2pt |
| Lens bins | 6 |
| Source bins | 4 |
| Bins per two-point observable | 30, 0.25–250 arcmin |
| Generated entries before cuts | 1,500 |
| Entries after the dataset mask | 635 |

The supplied file has 1,809 entries, but this generator currently computes only
its first 1,500 galaxy/shear entries. Their 635 retained entries form the
supported 3×2pt submatrix. CMB lensing auto- and cross-covariances are not
generated. The supplied full joint matrix remains available to the likelihood.

The default [installation options](../../set_installation_options.sh) set
`IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE=1`. This leaves covariance-generation
kernels and notebook bindings out of the compiled interface. Likelihoods still
read and invert their supplied covariance matrices. The steps below enable
covariance generation for this build; comment out that export in
`set_installation_options.sh` to keep it enabled in later sessions.
Recompile after changing the option, then restart any running notebook kernel.

We assume Cocoa and this project are installed, users have run
`conda activate cocoa`, the shell is Bash, and the current folder is
`cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: enable covariance generation and compile the project interface.

    unset IGNORE_COSMOLIKE_DESXPLANCK_CODE
    unset IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE
    source ./projects/desy1xplanck/scripts/compile_desy1xplanck.sh

**Step :three:**: start Jupyter.

    jupyter notebook --no-browser --port=8888

**Step :four:**: open the printed URL and select
`projects/desy1xplanck/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

**Step :five:**: inspect the survey inputs and keep `boosts = [1]` for the
first calculation, then select **Kernel → Restart Kernel and Run All Cells**.
Set `boosts = [1, 2]` to add the accuracy comparison.

The notebook writes `covariance/forecast_real.npz`,
`covariance/forecast_camb.npz` and
`covariance/forecast_likelihood_selection.npz`. The last archive retains
both cut totals and the original data-vector indices.
Set `spaces = ["real", "fourier"]` to compute both transformations; only the native space is compared with the supplied likelihood.
The [covariance guide](covariance/README.md) describes the physical inputs,
component plots, accuracy controls and covariance-only tests.

> [!NOTE]
> The generated matrix is an analogous forecast, not a reproduction of the
> supplied likelihood covariance. Gaussian spectra can include non-Limber
> gg/gs and NLA/TATT; SSC/cNG retain zero-IA Limber physics. The forecast
> uses massless neutrinos and a spherical-cap footprint.
> `accuracy_boost` refines
> tables and cutoffs; `integration_accuracy` separately selects precomputed
> GSL rules from [covariance/default.yaml](covariance/default.yaml).

## Command-line calculation

The Python runner computes the full galaxy–shear covariance in real space,
using the optimized production interface. It saves G, SSC, cNG and their
sum without plotting or opening a notebook. Numerical kernels and survey
settings are shared with the notebook calculation.

From Bash in `cocoa/Cocoa`, with `conda activate cocoa`:

**Step :one:**: activate Cocoa and enable covariance generation.

    source start_cocoa.sh
    unset IGNORE_COSMOLIKE_DESXPLANCK_CODE
    unset IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE

**Step :two:**: compile the project interface.

    source ./projects/desy1xplanck/scripts/compile_desy1xplanck.sh

**Step :three:**: inspect the YAML cosmology and compute the matrix components.

    export OMP_NUM_THREADS=8
    python ./projects/desy1xplanck/covariance/compute_covariance.py \
        ./projects/desy1xplanck/EXAMPLE_EVALUATE_COVARIANCE.yaml

The `.npz` archive contains the full matrix before likelihood scale cuts,
its components, measurement ordering, resolved settings and stage timings.
Existing output files require `--overwrite`; likelihood inputs are separate.

Set `covariance.space` to `real` or `fourier` to select the measurement.

The [evaluate YAML](EXAMPLE_EVALUATE_COVARIANCE.yaml) uses Cobaya's YAML reader, with familiar
`theory`, `params`, `sampler: evaluate` and `output` blocks. Fixed parameter
values specify one cosmology; a parameter with a prior must be supplied
explicitly in `sampler.evaluate.override`. No MCMC or random prior draw runs.

In its `covariance` block, `accuracy_boost: 2` refines the project's
`default.yaml` baseline. `integration_accuracy: 1` changes the quadrature
level independently. Internal accuracy controls can also be set there.
Use `space` for the measurement space. Set the OpenMP team with
`OMP_NUM_THREADS` in the shell; no thread count belongs in the YAML.

`theory.camb.extra_args` supports `AccuracyBoost`, `kmax`, `k_per_logint`,
`lens_potential_accuracy` and `halofit_version`. CAMB's boost controls
CAMB; the covariance boost controls its own tables and cutoffs.

Paths in the YAML are relative to the working directory, `cocoa/Cocoa`.
`output` names the `.npz` archive; `--output` can override it for an HPC
job. Set `OMP_NUM_THREADS` in that job’s environment. `--help` lists the
command options.

To return to a data-vector-only build, use the following steps from
`cocoa/Cocoa` with `conda activate cocoa` and Bash.

**Step :one:**: activate Cocoa.

    source start_cocoa.sh

**Step :two:**: omit covariance generation and rebuild the interface.

    unset IGNORE_COSMOLIKE_DESXPLANCK_CODE
    export IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE=1
    source ./projects/desy1xplanck/scripts/compile_desy1xplanck.sh

Gaussian non-Limber and NLA/TATT options are documented in the
[covariance guide](covariance/README.md#choosing-the-gaussian-spectra).
The YAML keeps these Gaussian choices separate from SSC/cNG. OpenMP
threads come exclusively from `OMP_NUM_THREADS`, not from a YAML key.
