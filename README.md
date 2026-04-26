# cocoa_desy1xplanck
DESY1 x Planck with COCOA

This project perform the 6x2pt joint-likelihood analyses in **mix space**, i.e. the 5x2pt twopcfs are measured in real space and the CMB lensing twopcf is measured in Fourier space 

Note: the CMB lensing reconstruction noise file is in the format of "ell C_ell^{dd}". To translate to C^{kk}\_ell, need to multiply ell(ell+1)/4

## CMB kappa debugging

To compare the data vector predictions from `cosmolike_core` and `cocoa`, run

> $ mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=4 cobaya-run ./projects/desy1xplanck/yaml/EXAMPLE_EVALUATE1.yaml -f

Or
> $ mpirun -n 1 --mca btl tcp,self cobaya-run ./projects/desy1xplanck/yaml/Y3xPLKR4/EVALUATE1.yaml -f


# WARNING: WHEN GENERATING PCs OF BARYON IMPACTS, PLEASE GENERATE WITH SHEAR-ONLY LIKELIHOOD. CURRENTLY WE ARE ONLY CONFIDENT TO APPLY BARYON PCS TO SHEAR-SHEAR CORRELATION FUNCTION, NOT THE OTHER PROBES.

The fiducial `cosmolike_core` data vector is `./data/xi_desy1xplanck_6x2pt_fid_cosmolike_core`
The fiducial parameters to generate `cosmolike_core` data vector is recorded in `./data/README.md`
The `./likelihood/desy1xplanck_6x2pt.py` will write the data vector at fiducial parameters to `./chains/EXAMPLE_EVALUATE1.model_vector`

## Running Cosmolike projects (Basic instructions) <a name="desy1xplanck_running_cosmolike_projects"></a> 

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
>     #export IGNORE_COSMOLIKE_LSSTY1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     (...)
>     export IGNORE_COSMOLIKE_DESXPLANCK_CODE=1
>
>     (...)
> 
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export ROMAN_REAL_URL="https://git@github.com/CosmoLike/cocoa_desy1xplanck.git"
>     export DESXPLANCK_GIT_NAME="desy1xplanck"
>     #BRANCH: if unset, load the latest commit on the specified branch
>     #export DESXPLANCK_GIT_BRANCH="main"
>     #COMMIT: if unset, load the specified commit
>     export DESXPLANCK_GIT_COMMIT="abc"
>     #BRANCH: if unset, load the specified TAG
>     export DESXPLANCK_GIT_TAG=v4.07

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

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment 
    
      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).
    
    export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE
      
 **Step :three:**: The folder `projects/desy1xplanck` contains examples. So, run the `cobaya-run` on the first example following the commands below.

- **One model evaluation**:

  - Linux

        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/desy1xplanck/EXAMPLE_EVALUATE1.yaml -f

  -  macOS (arm)

         mpirun -n 1 --oversubscribe cobaya-run ./projects/desy1xplanck/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/desy1xplanck/EXAMPLE_MCMC1.yaml -f

   -  macOS (arm)
     
          mpirun -n 4 --oversubscribe cobaya-run ./projects/desy1xplanck/EXAMPLE_MCMC1.yaml -f
