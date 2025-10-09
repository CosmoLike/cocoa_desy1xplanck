from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class combo_6x2pt(_cosmolike_prototype_base):
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def initialize(self):
    super(combo_6x2pt, self).initialize(probe="6x2pt")