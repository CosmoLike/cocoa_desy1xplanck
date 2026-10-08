"""The CMB-lensing blocks (cobaya name desy1xplanck.combo_3x2pt_ks_gk_kk).

This likelihood combines the three two-point functions that involve the
Planck CMB lensing convergence kappa: its cross-correlation with the
lens galaxy density (block gk) and with the source shear (block ks), and
its auto-spectrum bandpowers (block kk). Cosmic shear, galaxy-galaxy
lensing and galaxy clustering are left out. cobaya loads the class below
when a yaml file names desy1xplanck.combo_3x2pt_ks_gk_kk; its default
options are in combo_3x2pt_ks_gk_kk.yaml next to this file, which
includes params_source.yaml and params_lens_maglim.yaml and fixes, in
fixed_params, the point masses (they act on galaxy-galaxy lensing only)
and the parameters that act only on lens bins 5-6 (the mask removes
those bins; see likelihood/README.md). Everything else is inherited from
_cosmolike_prototype_base.
"""
# cobaya sees this folder as the package cobaya.likelihoods.desy1xplanck
# (Cocoa links projects/desy1xplanck/likelihood into cobaya's likelihoods
# folder), hence the import path below.
from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class combo_3x2pt_ks_gk_kk(_cosmolike_prototype_base):
 """Galaxy x CMB lensing (gk), shear x CMB lensing (ks) and kk bandpowers."""
 def initialize(self):
    """Configure cosmolike for the gk, ks and kk blocks.

    cobaya calls initialize() once, when it builds the likelihood.
    super(combo_3x2pt_ks_gk_kk, self) finds the parent class, so the line
    below runs _cosmolike_prototype_base.initialize with the probe name
    "3x2pt_ks_gk_kk".
    """
    super(combo_3x2pt_ks_gk_kk, self).initialize(probe="3x2pt_ks_gk_kk")