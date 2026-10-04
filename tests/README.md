# Tests

The two test sectors answer different questions and can be run separately.

- [Data-vector and likelihood tests](data_vector/README.md) check predicted
  signals, frozen likelihood values, numerical accuracy and repeated
  evaluations. This is the usual choice for likelihood users.
- [Covariance tests](covariance/README.md) check this project's forecast inputs, real/Fourier
  assembly, thread repeatability, positivity and saved-output metadata.

From `cocoa/Cocoa`, with the Cocoa environment active and `start_cocoa.sh`
available, follow these steps.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run the data-vector tests.

    python -m pytest projects/desy1xplanck/tests/data_vector

For covariance checks, we assume the Conda Cocoa environment, Bash,
and the current folder `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run the covariance tests.

    python -m pytest projects/desy1xplanck/tests/covariance

To check both sectors, we assume the Conda Cocoa environment, Bash,
and the current folder `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: select both test folders.

    python -m pytest projects/desy1xplanck/tests/data_vector projects/desy1xplanck/tests/covariance

The shared data-vector harness, frozen snapshots, their fingerprint
manifest and reference-generation scripts remain here in `tests/`.
Moving test modules does not change those snapshots or refreeze any result.
Covariance tests do not replace the likelihood's stored covariance.
