"""Shear plus shear x CMB lensing (cobaya name desy1xplanck.combo_2x2pt_ss_sk).

This likelihood combines cosmic shear (block ss, xi_+ and xi_- of the 4
source bins) and the cross-correlation of the source shear with the
Planck CMB lensing convergence (block ks, one angular correlation per
source bin). No lens galaxies enter. cobaya loads the class below when a
yaml file names desy1xplanck.combo_2x2pt_ss_sk; its default options are
in combo_2x2pt_ss_sk.yaml next to this file, which includes
params_source.yaml only. Everything else is inherited from
_cosmolike_prototype_base.
"""
# cobaya sees this folder as the package cobaya.likelihoods.desy1xplanck
# (Cocoa links projects/desy1xplanck/likelihood into cobaya's likelihoods
# folder), hence the import path below.
from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class combo_2x2pt_ss_sk(_cosmolike_prototype_base):
  """Cosmic shear (ss) and shear x CMB lensing (ks)."""
  def initialize(self):
    """Configure cosmolike for the ss and ks blocks (probe "2x2pt_ss_sk").

    cobaya calls initialize() once, when it builds the likelihood.
    super(combo_2x2pt_ss_sk, self) finds the parent class, so the line
    below runs _cosmolike_prototype_base.initialize with this probe name.
    """
    super(combo_2x2pt_ss_sk, self).initialize(probe="2x2pt_ss_sk")