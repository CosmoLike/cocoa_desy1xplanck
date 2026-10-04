# Covariance tests

These checks exercise this project's galaxy/shear forecast adapter and
compiled covariance bindings. They use the project's redshift files and
catalog inputs, with small numerical grids and three measured rows.

We assume Cocoa is installed, users have run `conda activate cocoa`, the
shell is Bash and the current folder is `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile this project's interface.

    unset IGNORE_COSMOLIKE_DESY1XPLANCK_CODE
    source projects/desy1xplanck/scripts/compile_desy1xplanck.sh

**Step :three:**: run the covariance checks.

    python -m pytest projects/desy1xplanck/tests/covariance

| Check | Purpose |
| --- | --- |
| Project layout | Verify the full angular and Fourier vector lengths from the measured row map. |
| Accuracy refinement | Keep measurement bins fixed while increasing quadrature resolution. |
| Real-space components | Check finite, symmetric G, SSC, cNG and total matrices for a measured subset. |
| Fourier components | Check the same properties for bandpowers. |
| Thread repeatability | Compare every component bitwise with one and eight OpenMP threads. |
| Positive total | Check variance positivity for every direction of the tested subset. |
| Output archive | Read arrays and resolved survey metadata without pickle. |

The [shared component tests](../../../lsst_y1/tests/covariance/README.md)
contain independent algebra and projection references. This project check
covers its binding and inputs; it does not duplicate those references.
Use the [covariance notebook](../../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
for full matrices and numerical refinement. Subset positivity alone does
not establish full-matrix positivity or parameter-error convergence.

The [data-vector tests](../data_vector/README.md) check likelihood signals
and stored reference values separately. No stored covariance or reference
value is replaced by this test.
