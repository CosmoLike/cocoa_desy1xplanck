"""Cosmic-shear likelihood of desy1xplanck (cobaya name desy1xplanck.cosmic_shear).

Cosmic shear (block ss of the data vector) is the correlation of galaxy
shapes distorted by weak gravitational lensing: xi_+ and xi_- of the 4
source redshift bins. cobaya loads the class below when a yaml file
names desy1xplanck.cosmic_shear; its default options are in
cosmic_shear.yaml next to this file, which includes the parameter file
params_source.yaml. Everything else is inherited from
_cosmolike_prototype_base, which reads the data, configures cosmolike
and computes the chi2.
"""
# cobaya sees this folder as the package cobaya.likelihoods.desy1xplanck
# (Cocoa links projects/desy1xplanck/likelihood into cobaya's likelihoods
# folder), hence the import path below.
from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  """The cosmic-shear block (ss) alone."""
  def initialize(self):
    """Configure cosmolike for cosmic shear only (probe "xi").

    cobaya calls initialize() once, when it builds the likelihood.
    super(cosmic_shear, self) finds the parent class, so the line below
    runs _cosmolike_prototype_base.initialize with this probe name.
    """
    super(cosmic_shear,self).initialize(probe="xi")