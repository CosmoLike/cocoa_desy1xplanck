from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class desy1xplanck_6x2pt(_cosmolike_prototype_base):
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def initialize(self):
    super(desy1xplanck_6x2pt,self).initialize(probe="6x2pt")

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params_values):
    if self.create_baryon_pca:
      self.generate_baryonic_PCA(**params_values)
      self.force_cache_false = True

    self.set_cosmo_related()

    if self.create_baryon_pca:
      self.force_cache_false = False

    self.set_lens_related(**params_values)

    self.set_source_related(**params_values)

    # datavector C++ returns a list (not numpy array)
    datavector = np.array(ci.compute_data_vector_masked())

    if self.use_baryon_pca:
      # Warning: we assume the PCs were created with the same mask
      # We have no way of testing user enforced that
      #print("\n\n\n ??? \n\n\n")
      self.set_baryon_related(**params_values)
      datavector = self.add_baryon_pcs_to_datavector(datavector)
    #print("Evaluated data vector:\n")
    dv_fname = "./projects/desy1xplanck/chains/EXAMPLE_EVALUATE1.model_vector"
    with open(dv_fname, 'w') as fp:
        for i in range(len(datavector)):
            #print("i=%d dv = %le\n"%(i,datavector[i]))
            fp.write("%d %le\n"%(i, datavector[i]))
    return self.compute_logp(datavector)

