"""6x2pt likelihood of desy1xplanck (cobaya name desy1xplanck.combo_6x2pt).

6x2pt is the joint analysis of six two-point functions: the three of the
galaxy survey, cosmic shear (block ss, xi_+ and xi_-), galaxy-galaxy
lensing (block gs, gamma_t) and galaxy clustering (block gg, w(theta)),
plus the three that involve the Planck CMB lensing convergence kappa:
its cross-correlations with the lens galaxy density (block gk) and with
the source shear (block ks), and its auto-spectrum bandpowers (block
kk). cobaya loads the class below when a yaml file names
desy1xplanck.combo_6x2pt; its default options are in combo_6x2pt.yaml
next to this file, which includes params_source.yaml and
params_lens_maglim.yaml and fixes, in fixed_params, the parameters that
act only on lens bins 5-6 (the mask removes those bins; see
likelihood/README.md). Everything else is inherited from
_cosmolike_prototype_base.
"""
# cobaya sees this folder as the package cobaya.likelihoods.desy1xplanck
# (Cocoa links projects/desy1xplanck/likelihood into cobaya's likelihoods
# folder), hence the import path below.
from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class combo_6x2pt(_cosmolike_prototype_base):
  """All six blocks: ss, gs, gg, gk, ks and kk."""
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def initialize(self):
    """Configure cosmolike for every block of the data vector ("6x2pt").

    cobaya calls initialize() once, when it builds the likelihood.
    super(combo_6x2pt, self) finds the parent class, so the line below
    runs _cosmolike_prototype_base.initialize with this probe name.
    """
    super(combo_6x2pt, self).initialize(probe="6x2pt")