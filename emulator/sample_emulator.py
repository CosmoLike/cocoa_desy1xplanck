import sys
import os
from os.path import join as pjoin
import numpy as np
import torch
from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler
import emcee

### Parallelization
#from multiprocessing import Pool
from schwimmbad import MPIPool

### This file use pre-trained emulator to run MCMC chains based on input YAML
### Usage: ${PYTHON3} sample_emulator.py ${CONFIG}

if __name__ == '__main__':
	configfile = sys.argv[1]
	config = Config(configfile)
	# read emulators
	probe_fmts = ['xi_p', 'xi_m', 'gammat', 'wtheta', 'gk', 'ks', 'kk']
	probe_N = [config.N_xi, config.N_xi, config.N_ggl, config.N_w, config.N_gk, config.N_sk, config.N_kk]
	Niter = config.n_train_iter
	emu_list = []
	N_count = 0
	if (config.emu_type.lower()=='nn'):
		for i,p in enumerate(probe_fmts):
			_l, _r = N_count, N_count + probe_N[i]
			fn = pjoin(config.modeldir, f'{p}_{Niter-1}_nn{config.nn_model}')
			if os.path.exists(fn+".h5"):
				print(f'Reading {p} NN emulator from {fn}.h5 ...')
				emu = NNEmulator(config.n_dim, probe_N[i], config.dv_fid[_l:_r],
					config.dv_std[_l:_r], config.mask[_l:_r], config.nn_model)
				emu.load(fn)
			else:
				print(f'Can not find {p} emulator {fn}! Ignore probe {p}!')
				emu = None
			N_count += probe_N[i]
			emu_list.append(emu)
	elif (config.emu_type.lower()=='gp'):
		for i,p in enumerate(probe_fmts):
			_l, _r = N_count, N_count + probe_N[i]
			fn = pjoin(config.modeldir, f'{p}_{Niter-1}_gp')
			if os.path.exists(fn+".h5"):
				print(f'Reading {p} GP emulator from {fn}.h5 ...')
				emu = GPEmulator(config.n_dim, probe_N[i], config.dv_fid[_l:_r],
					config.dv_std[_l:_r])
				emu.load(fn)
			else:
				print(f'Can not find {p} emulator {fn}! Ignore probe {p}!')
				emu = None
			N_count += probe_N[i]
			emu_list.append(emu)
	else:
		print(f'emulator {config.emu_type} is not implemented!')
		exit(-1)

	# read sigma8 emulator
	if (config.emu_type.lower()=='nn'):
		fn = pjoin(config.modeldir, f'sigma8_{Niter-1}_nn{config.nn_model}')
		if os.path.exists(fn+".h5"):
			print(f'Reading sigma8 NN emulator from {fn}.h5 ...')
			emu_s8 = NNEmulator(config.n_pars_cosmo, 1, config.sigma8_fid,
					config.sigma8_std, np.array([True,]), config.nn_model)
			emu_s8.load(fn)
		else:
			print(f'Can not find sigma8 emulator {fn}!')
			emu_s8 = None
	elif (config.emu_type.lower()=='gp'):
		fn = pjoin(config.modeldir, f'sigma8_{Niter-1}_gp')
		if os.path.exists(fn+".h5"):
			print(f'Reading sigma8 GP emulator from {fn}.h5 ...')
			emu_s8 = GPEmulator(config.n_pars_cosmo, 1, config.sigma8_fid,
					config.sigma8_std)
			emu_s8.load(fn)
		else:
			print(f'Can not find sigma8 emulator {fn}!')
			emu_s8 = None


	os.environ["OMP_NUM_THREADS"] = "1"
	emu_sampler = EmuSampler(emu_list, config)
	pos0 = emu_sampler.get_starting_pos()

	def ln_prob_wrapper(theta, temper=1.0):
		return emu_sampler.ln_prob(theta, temper)

	with MPIPool() as pool:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)
		sampler = emcee.EnsembleSampler(config.n_emcee_walkers, emu_sampler.n_sample_dims, ln_prob_wrapper, pool=pool)
		sampler.run_mcmc(pos0, config.n_mcmc, progress=True)

	samples = sampler.get_chain(discard=config.n_burn_in, thin=config.n_thin, flat=True)
	logprobs= sampler.get_log_prob(discard=config.n_burn_in, thin=config.n_thin, flat=True)

	if emu_s8 is not None:
		derived_sigma8 = emu_s8.predict(torch.Tensor(samples[:,:config.n_pars_cosmo]))
		np.save(pjoin(config.chaindir, config.chainname+'.npy'), 
				np.hstack([samples, derived_sigma8, logprobs[:,np.newaxis]]))
	else:
		np.save(pjoin(config.chaindir, config.chainname+'.npy'), 
			    np.hstack([samples, logprobs[:,np.newaxis]]))

