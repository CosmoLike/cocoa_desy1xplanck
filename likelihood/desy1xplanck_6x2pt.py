from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci

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
		self.set_cosmo_related()

		self.set_lens_related(**params_values)

		self.set_source_related(**params_values)

		datavector = ci.compute_data_vector()

		if self.print_intermediate_products == True:
			self.test_all()

		return self.compute_logp(datavector)

