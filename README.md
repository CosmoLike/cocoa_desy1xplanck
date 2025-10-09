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
