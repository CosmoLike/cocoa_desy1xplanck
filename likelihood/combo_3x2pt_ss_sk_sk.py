"""Shear and CMB lensing (cobaya name desy1xplanck.combo_3x2pt_ss_sk_sk).

This likelihood combines cosmic shear (block ss, xi_+ and xi_- of the 4
source bins), the cross-correlation of the source shear with the Planck
CMB lensing convergence (block ks) and the CMB lensing auto-spectrum
bandpowers (block kk). No lens galaxies enter. The probe name
"3x2pt_ss_sk_sk" repeats "sk", but cosmolike's probe table maps it to
ss + ks + kk. cobaya loads the class below when a yaml file names
desy1xplanck.combo_3x2pt_ss_sk_sk; its default options are in
combo_3x2pt_ss_sk_sk.yaml next to this file, which includes
params_source.yaml only. Everything else is inherited from
_cosmolike_prototype_base.
"""
# cobaya sees this folder as the package cobaya.likelihoods.desy1xplanck
# (Cocoa links projects/desy1xplanck/likelihood into cobaya's likelihoods
# folder), hence the import path below.
from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class combo_3x2pt_ss_sk_sk(_cosmolike_prototype_base):
 """Cosmic shear (ss), shear x CMB lensing (ks) and kk bandpowers."""
 def initialize(self):
    """Configure cosmolike for the ss, ks and kk blocks.

    cobaya calls initialize() once, when it builds the likelihood.
    super(combo_3x2pt_ss_sk_sk, self) finds the parent class, so the line
    below runs _cosmolike_prototype_base.initialize with the probe name
    "3x2pt_ss_sk_sk".
    """
    super(combo_3x2pt_ss_sk_sk, self).initialize(probe="3x2pt_ss_sk_sk")