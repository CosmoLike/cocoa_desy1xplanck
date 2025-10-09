from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class combo_2x2pt_ss_sg(_cosmolike_prototype_base):
  def initialize(self):
    super(combo_2x2pt_ss_sg, self).initialize(probe="2x2pt_ss_sg")
