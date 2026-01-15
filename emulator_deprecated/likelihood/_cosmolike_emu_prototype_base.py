import numpy as np
import os
from os.path import join as pjoin
from getdist import IniFile
from cobaya.likelihoods._base_classes import _DataSetLikelihood
import torch
from cocoa_emu import Config
from cocoa_emu.emulator import NNEmulator

probe_fmts = ["xi_pm", "gammat", "wtheta", "wgk", "wsk", "Ckk"]

#torch.set_default_dtype(torch.float64)

class _cosmolike_emu_prototype_base(_DataSetLikelihood):
	''' Attributes needed from the likelihood yaml file:
	- train_config: filename of the training config file
	'''
	def initialize(self, probe):
		super(_cosmolike_emu_prototype_base, self)
		torch.set_num_threads(1)
		self.device = torch.device("cpu")
		self.probe = probe

		# Note: This config file is only used to calculate data vector
		#       The baryon PCs and priors are overwritten during sampling
		self.log.info(f'Init emulator with config file {self.train_config}')
		config = Config(self.train_config)
		assert config.emu_type.lower()=='nn', f'Only support NN emulator now!'
		self.probe_mask        = config.probe_mask_choices[self.probe]
		self.probe_size        = config.probe_size

		### Read dataset file: data vector, covariance, mask
		self.log.info("Loading likelihood dataset...")
		ini = IniFile(os.path.normpath(pjoin(self.path, self.data_file)))
		self.data_vector_file = ini.relativeFileName('data_file')
		self.cov_file = ini.relativeFileName('cov_file')
		try:
			self.U_PMmarg_file = ini.relativeFileName('U_PMmarg')
		except:
			raise LoggedError(self.log, "Can not find Point-Mass analytical marginalization matrix, go without it.")
			self.U_PMmarg_file = ""
		self.mask_file = ini.relativeFileName('mask_file')
		self.source_ntomo = ini.int("source_ntomo")
		self.lens_ntomo = ini.int("lens_ntomo", default = -1)
		self.ntheta = ini.int("n_theta")
		self.nbp = ini.int("n_bp")
		self.dv_size = self.ntheta*(self.source_ntomo*(self.source_ntomo+1)+\
			self.source_ntomo*self.lens_ntomo + self.lens_ntomo+\
			self.lens_ntomo + self.source_ntomo) + self.nbp
		# CMB lensing covariance specs
		self.is_cmbl_cov_sim = ini.int("is_cmb_kkkk_cov_from_sim", default = -1)
		if(self.is_cmbl_cov_sim == 1):
			Nvar = ini.float("Hartlap_Nvar")
			self.alpha_Hartlap = (Nvar - self.nbp -2.0)/(Nvar - 1.0) # < 1
		elif(self.is_cmbl_cov_sim==-1 or self.is_cmbl_cov_sim > 1):
			raise LoggedError(self.log, "MUST SPECIFY is_cmb_kkkk_cov_from_sim (0 or 1) IN THE DATA FILE!")
		else:
			self.is_cmbl_cov_sim = 0
			self.alpha_Hartlap = 1.0
		self.init_data(config)

		### Initialize baryon feedback PCs
		if ini.string('baryon_pca_file', default=''):
			baryon_pca_file = ini.relativeFileName('baryon_pca_file')
			self.baryon_pcs = np.loadtxt(baryon_pca_file)
			self.log.info('use_baryon_pca = True')
			self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
			self.use_baryon_pca = True
			if self.subtract_mean:
				mean_baryon_diff_file = ini.relativeFileName('mean_baryon_diff_file')
				self.mean_baryon_diff = np.loadtxt(mean_baryon_diff_file)
				self.log.info('subtract_mean = True')
				self.log.info('mean_baryon_diff_file = %s loaded', mean_baryon_diff_file)
			else:
				self.log.info('subtract_mean = False')
		else:
			self.log.info('use_baryon_pca = False')
			self.use_baryon_pca = False
		self.baryon_pcs_qs = np.zeros(4)

		self.shear_calib_mask  = config.shear_calib_mask
		self.n_pars_cosmo      = config.n_pars_cosmo
		self.running_params    = config.running_params
		self.m_shear_fid       = np.array([config.params["DES_M%d"%(i+1)]["value"] for i in range(self.source_ntomo)])

		### read emulators
		# try include emu_list as object attribute. If not work, global variable
		self.log.info("Reading emulator models...")
		self.log.info(f'invcov dtype: {config.inv_cov.dtype}')
		self.emu_list = []
		for i,p in enumerate(probe_fmts):
			_l, _r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
			fn = pjoin(config.modeldir, f'{p}_nn{config.nn_model}')
			if os.path.exists(fn+".h5"):
				self.log.info(f'--- Reading {p} NN emulator from {fn}.h5 ...')
				emu = NNEmulator(config.n_dim, config.probe_size[i], 
					config.dv_lkl[_l:_r], config.dv_std[_l:_r],
					config.inv_cov[_l:_r,_l:_r],
					mask=config.mask_lkl[_l:_r],
					param_mask=config.probe_params_mask[i],
					model=config.nn_model, device=self.device,
					deproj_PCA=True, lr=config.learning_rate, 
					reduce_lr=config.reduce_lr, 
					weight_decay=config.weight_decay, dtype="double")
				emu.load(fn)
			else:
				self.log.info(f'{fn} not found! Ignore probe {p} emulator!')
				emu = None
			self.emu_list.append(emu)
		# read sigma8 emulator
		fn = pjoin(config.modeldir, f'sigma8_nn{config.nn_model}')
		if os.path.exists(fn+".h5"):
			self.log.info(f'--- Reading sigma8 NN emulator from {fn}.h5 ...')
			self.emu_s8 = NNEmulator(config.n_pars_cosmo, 1, config.sigma8_fid, 
					config.sigma8_std, np.atleast_2d(1.0/config.sigma8_std**2), 
					model=config.nn_model, device=self.device,
					deproj_PCA=False, lr=config.learning_rate, 
					reduce_lr=config.reduce_lr, 
					weight_decay=config.weight_decay, dtype="double")
			self.emu_s8.load(fn)
		else:
			self.log.info(f'{fn} not found! Ignore sigma8 emulator!')
			self.emu_s8 = None
		self.log.info("Emulator likelihood initialized!")

	def init_data(self, config):
		''' Prepare the likelihood dataset
		Including inverse covariance, data vector, data vector mask
		Equivalent to `ci.init_data`
		'''
		### prepare data vector & mask
		self.log.info(f'Load data vector from {self.data_vector_file}')
		self.dv   = np.loadtxt(self.data_vector_file)[:,1]
		self.log.info(f'Load mask from {self.mask_file}')
		self.mask = np.loadtxt(self.mask_file)[:,1].astype(bool)
		# update the mask if some probes are not included
		for i in range(6):
			_l, _r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
			if self.probe_mask[i]==0:
				self.mask[_l:_r] = 0.0
				self.log.info(f'Probe {probe_fmts[i]} is not included.')
			else:
				_Ndp = self.mask[_l:_r].sum()
				self.log.info(f'Probe {probe_fmts[i]} has {_Ndp} elements after scale cut.')
		self.log.info(f'Total data points: {sum(self.mask)}')
		### prepare inverse covariance
		self.log.info(f'Load covariance from {self.cov_file}')
		invcov = self.get_full_cov(self.cov_file)
		# Add Hartlap factor to CMB lensing covariance
		self.log.info(f'Apply Hartlap factor {self.alpha_Hartlap}')
		invcov[-self.nbp:,-self.nbp:] /= self.alpha_Hartlap
		invcov = np.linalg.inv(invcov[self.mask][:,self.mask])
		# Add PM marginalization
		if self.U_PMmarg_file:
			self.log.info(f'Load PM-marg template from {self.U_PMmarg_file}')
			U_PMmarg = np.loadtxt(self.U_PMmarg_file)
			U = np.zeros([self.mask.shape[0], self.lens_ntomo])
			for line in U_PMmarg:
				i, j = int(line[0]), int(line[1])
				U[i,j] = float(line[2])
			U = U[self.mask,:]
			central_block = np.diag(np.ones(self.lens_ntomo)) + U.T@invcov@U
			w, v = np.linalg.eig(central_block)
			assert np.min(w)>=0, f'Central block not positive-definite!'
			corr = invcov @ (U@np.linalg.inv(central_block)@U.T) @ invcov
			invcov -= corr
		self.masked_inv_cov = invcov
		# test positive-definite
		w, v = np.linalg.eig(self.masked_inv_cov)
		assert np.min(w)>=0, f'Precision matrix not positive-definite!'

	def get_full_cov(self, cov_file):
		full_cov = np.loadtxt(cov_file)
		cov = np.zeros((self.dv_size, self.dv_size))
		cov_scenario = full_cov.shape[1]
		
		for line in full_cov:
			i = int(line[0])
			j = int(line[1])

			if(cov_scenario==3):
				cov_ij = line[2]
			elif(cov_scenario==10):
				cov_g_block  = line[8]
				cov_ng_block = line[9]
				cov_ij = cov_g_block + cov_ng_block
			cov[i,j] = cov_ij
			cov[j,i] = cov_ij
		return cov

	def emu_predict(self, theta):
		''' Get the emulator prediction for slow parameters
		'''
		theta = torch.Tensor(theta)
		# evaluate data vector using list of emulators
		model_vectors = []
		for i in range(6):
			if self.probe_mask[i]==1:
				_mv = self.emu_list[i].predict(theta)[0]
			else:
				_mv = np.zeros(self.probe_size[i])
			model_vectors.append(_mv)
		modelvector = np.hstack(model_vectors)
		return modelvector

	def get_sigma8_emu(self, **params_values):
		theta = np.array([params_values.get(p, 0.0) for p in self.running_params])
		theta = torch.Tensor(theta[:self.n_pars_cosmo])
		sigma8 = self.emu_s8.predict(theta)[0]
		return sigma8

	def get_model_vector_emu(self, **params_values):
		''' Evaluate model vector given parsed input sampled parameter array
		Note that linear galaxy bias are slow parameters due to magnification
		bias and RSD.
		'''
		# get model vector from emulated parameters
		theta = np.array([params_values.get(p, 0.0) for p in self.running_params])
		mv = self.emu_predict(theta)

		# add shear calibration bias
		m = np.array([params_values.get(f'DES_M{i+1}', 0.0) for i in range(self.source_ntomo)])
		for i in range(self.source_ntomo):
			factor = ((1+m[i])/(1+self.m_shear_fid[i]))**self.shear_calib_mask[i]
			mv = factor * mv
		
		# add baryon feedback PCs
		if self.use_baryon_pca:
			# Warning: we assume the PCs were created with the same mask
			# We have no way of testing user enforced that
			self.set_baryon_related(**params_values)
			mv = self.add_baryon_pcs_to_datavector(mv)
		return mv

	def set_baryon_related(self, **params_values):
		self.baryon_pcs_qs[0] = params_values.get("DES_BARYON_Q1", 0.0)
		self.baryon_pcs_qs[1] = params_values.get("DES_BARYON_Q2", 0.0)
		self.baryon_pcs_qs[2] = params_values.get("DES_BARYON_Q3", 0.0)
		self.baryon_pcs_qs[3] = params_values.get("DES_BARYON_Q4", 0.0)

	def add_baryon_pcs_to_datavector(self, datavector):
		if self.subtract_mean:
			datavector[:] += self.mean_baryon_diff 
		return datavector[:] + self.baryon_pcs_qs[0]*self.baryon_pcs[:,0] \
		  + self.baryon_pcs_qs[1]*self.baryon_pcs[:,1] \
		  + self.baryon_pcs_qs[2]*self.baryon_pcs[:,2] \
		  + self.baryon_pcs_qs[3]*self.baryon_pcs[:,3]

	def logp(self, **params_values):
		''' Evaluate the log-posterior of the likelihood
		Input:
		======
		params_values: dict
			dictionary of sampled input parameters
		'''
		mv = self.get_model_vector_emu(**params_values)
		delta_dv = (mv - self.dv)[self.mask]
		log_p = -0.5 * delta_dv @ self.masked_inv_cov @ delta_dv

		# derived parameters: sigma8
		if self.derive_sigma8 and self.emu_s8:
			params_values["_derived"]['sigma8'] = self.get_sigma8_emu(**params_values)

		return log_p
