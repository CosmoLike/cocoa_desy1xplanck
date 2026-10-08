"""Shear plus galaxy-galaxy lensing (cobaya name desy1xplanck.combo_2x2pt_ss_sg).

This likelihood combines cosmic shear (block ss, xi_+ and xi_- of the
source galaxies) and galaxy-galaxy lensing (block gs, gamma_t: the shear
of source galaxies around lens galaxies). cobaya loads the class below
when a yaml file names desy1xplanck.combo_2x2pt_ss_sg; its default
options are in combo_2x2pt_ss_sg.yaml next to this file, which includes
params_source.yaml and params_lens_maglim.yaml and fixes, in
fixed_params, the parameters that act only on lens bins 5-6 (the mask
removes those bins; see likelihood/README.md). Everything else is
inherited from _cosmolike_prototype_base.
"""
# cobaya sees this folder as the package cobaya.likelihoods.desy1xplanck
# (Cocoa links projects/desy1xplanck/likelihood into cobaya's likelihoods
# folder), hence the import path below.
from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class combo_2x2pt_ss_sg(_cosmolike_prototype_base):
  """Cosmic shear (ss) and galaxy-galaxy lensing (gs)."""
  def initialize(self):
    """Configure cosmolike for the ss and gs blocks (probe "2x2pt_ss_sg").

    cobaya calls initialize() once, when it builds the likelihood.
    super(combo_2x2pt_ss_sg, self) finds the parent class, so the line
    below runs _cosmolike_prototype_base.initialize with this probe name.
    """
    super(combo_2x2pt_ss_sg, self).initialize(probe="2x2pt_ss_sg")
