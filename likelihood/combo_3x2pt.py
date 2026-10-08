"""3x2pt likelihood of desy1xplanck (cobaya name desy1xplanck.combo_3x2pt).

3x2pt combines the three two-point functions of the galaxy survey:
cosmic shear (block ss, xi_+ and xi_- of the source galaxies),
galaxy-galaxy lensing (block gs, gamma_t: the shear of source galaxies
around lens galaxies) and galaxy clustering (block gg, w(theta) of the
lens galaxies). cobaya loads the class below when a yaml file names
desy1xplanck.combo_3x2pt; its default options are in combo_3x2pt.yaml
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

class combo_3x2pt(_cosmolike_prototype_base):
  """Cosmic shear (ss), galaxy-galaxy lensing (gs) and clustering (gg)."""
  def initialize(self):
    """Configure cosmolike for the ss, gs and gg blocks (probe "3x2pt").

    cobaya calls initialize() once, when it builds the likelihood.
    super(combo_3x2pt, self) finds the parent class, so the line below
    runs _cosmolike_prototype_base.initialize with this probe name.
    """
    super(combo_3x2pt,self).initialize(probe="3x2pt")