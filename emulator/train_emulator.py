import sys
import os
from os.path import join as pjoin
import numpy as np
import torch
from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler
import emcee
from argparse import ArgumentParser
from multiprocessing import Pool

parser = ArgumentParser()
parser.add_argument('config', type=str, help='Configuration file')
parser.add_argument('iter', type=int, help='Training iteration')
parser.add_argument('--temper', action='store_true', default=False,
                    help='Turn on likelihood temperature')
parser.add_argument('--save_emu', action='store_true', default=False,
                    help='Save emulator model data set trained in this iteration')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Turn on debugging mode')
parser.add_argument('--load_train_if_exist', action='store_true', default=False,
                    help='Load existing model data set and skip training if exist')
parser.add_argument('--only_new_sample', action='store_true', 
    default=False, help='Only train on new data from this iteration.')
parser.add_argument('--no_retrain', action='store_true', default=False, 
    help='Do NOT retrain previous model, train a new model instead.')
args = parser.parse_args()

#===============================================================================
temper_schedule = [0.02, 0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 0.9]

config = Config(args.config)
n      = args.iter
if(args.temper):
    temper_val = temper_schedule[n]
else:
    temper_val = 1.
print(f'\n>>> Start Emulator Training [Iteration {args.iter}] [1/Temperature {temper_val:2.3f}]\n')
label = config.emu_type.lower()
if label=="nn":
    label = label+f'{config.nn_model}'
#================== Loading Training Data ======================================
if args.only_new_sample:
    print(f'Loading training data: ONLY LOADING DATA FROM ITERATION {n}!')
    train_samples = np.load(pjoin(config.traindir, f'samples_{label}_{n}.npy'))
    train_data_vectors = np.load(pjoin(config.traindir, 
        f'data_vectors_{label}_{n}.npy'))
    train_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label}_{n}.npy'))
else:
    print(f'Loading training data: LOADING ALL DATA UNTIL ITERATION {n}!')
    train_samples = np.load(pjoin(config.traindir, f'samples_{label}_{0}.npy'))
    train_data_vectors = np.load(pjoin(config.traindir, 
        f'data_vectors_{label}_{0}.npy'))
    train_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label}_{0}.npy'))
    for i in range(1, n+1):
        train_samples = np.vstack([train_samples, 
            np.load(pjoin(config.traindir, f'samples_{label}_{i}.npy'))])
        train_data_vectors = np.vstack([train_data_vectors,
            np.load(pjoin(config.traindir, f'data_vectors_{label}_{i}.npy'))])
        train_sigma8 = np.vstack([train_sigma8,
            np.load(pjoin(config.traindir, f'sigma8_{label}_{i}.npy'))])
#================= Clean data by chi2 cut ======================================

def get_chi_sq_cut(train_data_vectors):
    chi_sq_list = []
    for dv in train_data_vectors:
        delta_dv = (dv - config.dv_obs)[config.mask]
        chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
        chi_sq_list.append(chi_sq)
    chi_sq_arr = np.array(chi_sq_list)
    select_chi_sq = (chi_sq_arr < config.chi_sq_cut)
    return select_chi_sq

select_chi_sq = get_chi_sq_cut(train_data_vectors)
selected_obj = np.sum(select_chi_sq)
total_obj    = len(train_data_vectors)
print("Total number of objects: %d"%(selected_obj))

train_data_vectors = train_data_vectors[select_chi_sq]
train_samples      = train_samples[select_chi_sq]
train_sigma8       = train_sigma8[select_chi_sq]
#================= Init emulator ===============================================
torch.set_num_threads(48)
# NN only hard-coded?
full_emu = NNEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std, config.inv_cov, config.mask_ones, config.nn_model)
#================= Training emulator ===========================================
# switch according to probes
if (config.shear_shear==1):
    print("=======================================")
    _l, _r = 0, config.N_xi
    emu_xi_plus = NNEmulator(config.n_dim, config.N_xi, 
            config.dv_fid[_l:_r], config.dv_std[_l:_r], 
            config.inv_cov[_l:_r,_l:_r],
            config.mask[_l:_r], config.nn_model)
    emu_xi_plus_fn = pjoin(config.modeldir, f'xi_p_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_xi_plus_fn)):
        print(f'Loading existing xi_plus emulator from {emu_xi_plus_fn}....')
        emu_xi_plus.load(emu_xi_plus_fn)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'xi_p_{n-1}_nn{config.nn_model}')
            print(f'Retrain xi_plus emulator from {previous_fn}')
            emu_xi_plus.load(previous_fn)
        else:
            print("Training NEW xi_plus emulator....")
        emu_xi_plus.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs)
        if(args.save_emu):
            emu_xi_plus.save(emu_xi_plus_fn)
    print("=======================================")
    print("=======================================")
    _l, _r = config.N_xi, config.N_xi*2
    emu_xi_minus = NNEmulator(config.n_dim, config.N_xi, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        config.mask[_l:_r], config.nn_model)
    emu_xi_minus_fn = pjoin(config.modeldir, f'xi_m_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_xi_minus_fn)):
        print(f'Loading existing xi_minus emulator from {emu_xi_minus_fn}....')
        emu_xi_minus.load(emu_xi_minus_fn)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'xi_m_{n-1}_nn{config.nn_model}')
            print(f'Retrain xi_minus emulator from {previous_fn}')
            emu_xi_minus.load(previous_fn)
        else:
            print("Training NEW xi_minus emulator....")
        emu_xi_minus.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs)
        if(args.save_emu):
            emu_xi_minus.save(emu_xi_minus_fn)
    print("=======================================")
if (config.shear_pos==1):
    print("=======================================")
    _l, _r = config.N_xi*2, config.N_xi*2 + config.N_ggl
    emu_gammat = NNEmulator(config.n_dim, config.N_ggl, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        config.mask[_l:_r], config.nn_model)
    emu_gammat_fn = pjoin(config.modeldir, f'gammat_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_gammat_fn)):
        print(f'Loading existing gammat emulator from {emu_gammat_fn}....')
        emu_gammat.load(emu_gammat_fn)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'gammat_{n-1}_nn{config.nn_model}')
            print(f'Retrain gammat emulator from {previous_fn}')
            emu_gammat.load(previous_fn)
        else:
            print("Training NEW gammat emulator....")
        emu_gammat.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs)
        if(args.save_emu):
            emu_gammat.save(emu_gammat_fn)
    print("=======================================")
if (config.pos_pos==1):
    print("=======================================")
    _l, _r = config.N_xi*2+config.N_ggl, config.N_xi*2+config.N_ggl+config.N_w
    emu_wtheta = NNEmulator(config.n_dim, config.N_w, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        config.mask[_l:_r], config.nn_model)
    emu_wtheta_fn = pjoin(config.modeldir, f'wtheta_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_wtheta_fn)):
        print(f'Loading existing wtheta emulator from {emu_wtheta_fn}....')
        emu_wtheta.load(emu_wtheta_fn)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'wtheta_{n-1}_nn{config.nn_model}')
            print(f'Retrain wtheta emulator from {previous_fn}')
            emu_wtheta.load(previous_fn)
        else:
            print("Training NEW wtheta emulator....")
        emu_wtheta.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs)
        if(args.save_emu):
            emu_wtheta.save(emu_wtheta_fn)
    print("=======================================")
if (config.gk==1):
    print("=======================================")
    _l, _r = config.N_xi*2+config.N_ggl+config.N_w, config.N_xi*2+config.N_ggl+config.N_w+config.N_gk
    emu_gk = NNEmulator(config.n_dim, config.N_gk, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        config.mask[_l:_r], config.nn_model)
    emu_gk_fn = pjoin(config.modeldir, f'gk_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_gk_fn)):
        print(f'Loading existing w_gk emulator from {emu_gk_fn}....')
        emu_gk.load(emu_gk_fn)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'gk_{n-1}_nn{config.nn_model}')
            print(f'Retrain w_gk emulator from {previous_fn}')
            emu_gk.load(previous_fn)
        else:
            print("Training NEW w_gk emulator....")
        emu_gk.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs)
        if(args.save_emu):
            emu_gk.save(emu_gk_fn)
    print("=======================================")
if (config.ks==1):
    print("=======================================")
    _l, _r = config.N_xi*2+config.N_ggl+config.N_w+config.N_gk, config.N_xi*2+config.N_ggl+config.N_w+config.N_gk+config.N_sk
    emu_ks = NNEmulator(config.n_dim, config.N_sk, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        config.mask[_l:_r], config.nn_model)
    emu_ks_fn = pjoin(config.modeldir, f'ks_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_ks_fn)):
        print(f'Loading existing w_sk emulator from {emu_ks_fn}....')
        emu_ks.load(emu_ks_fn)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'ks_{n-1}_nn{config.nn_model}')
            print(f'Retrain w_sk emulator from {previous_fn}')
            emu_ks.load(previous_fn)
        else:
            print("Training NEW w_sk emulator....")
        emu_ks.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs)
        if(args.save_emu):
            emu_ks.save(emu_ks_fn)
    print("=======================================")
if (config.kk==1):
    print("=======================================")
    _l, _r = config.N_xi*2+config.N_ggl+config.N_w+config.N_gk+config.N_sk, config.N_xi*2+config.N_ggl+config.N_w+config.N_gk+config.N_sk+config.N_kk
    emu_kk = NNEmulator(config.n_dim, config.N_kk, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.inv_cov[_l:_r,_l:_r],
        config.mask[_l:_r], config.nn_model)
    emu_kk_fn = pjoin(config.modeldir, f'kk_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_kk_fn)):
        print(f'Loading existing CMBL band power emulator from {emu_kk_fn}....')
        emu_kk.load(emu_kk_fn)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'kk_{n-1}_nn{config.nn_model}')
            print(f'Retrain CMBL band power emulator from {previous_fn}')
            emu_kk.load(previous_fn)
        else:
            print("Training NEW CMBL band power emulator....")
        emu_kk.train(torch.Tensor(train_samples), 
            torch.Tensor(train_data_vectors[:,_l:_r]),
            batch_size=config.batch_size, n_epochs=config.n_epochs)
        if(args.save_emu):
            emu_kk.save(emu_kk_fn)
    print("=======================================")
if (config.derived==1):
    print("=======================================")
    emu_s8 = NNEmulator(config.n_pars_cosmo, 1, 
        config.sigma8_fid, config.sigma8_std, 1.0/config.sigma8_fid**2, 
        np.array([True,]), config.nn_model)
    emu_s8_fn = pjoin(config.modeldir, f'sigma8_{n}_nn{config.nn_model}')
    if (args.load_train_if_exist and os.path.exists(emu_s8_fn)):
        print(f'Loading existing derived parameters emulator (sigma8) from {emu_s8_fn}....')
        emu_s8.load(emu_s8_fn)
    else:
        if ((not args.no_retrain) and (n>0)):
            previous_fn = pjoin(config.modeldir, 
                f'sigma8_{n-1}_nn{config.nn_model}')
            print(f'Retrain derived parameters emulator from {previous_fn}')
            emu_s8.load(previous_fn)
        else:
            print("Training NEW derived parameters emulator....")
        emu_s8.train(torch.Tensor(train_samples[:,:config.n_pars_cosmo]), 
            torch.Tensor(train_sigma8),
            batch_size=config.batch_size, n_epochs=config.n_epochs)
        if(args.save_emu):
            emu_s8.save(emu_s8_fn)
    print("=======================================")
#==============================================
os.environ["OMP_NUM_THREADS"] = "1"

emu_sampler = EmuSampler(full_emu, config)
pos0 = emu_sampler.get_starting_pos()
#==============================================
self = emu_sampler

def compute_datavector(theta, emu):
    theta = torch.Tensor(theta)
    datavector = emu.predict(theta)[0]        
    return datavector
    
def get_data_vector_emu(theta):
    theta_emu   = theta[:-self.n_fast_pars]
    
    if (config.shear_shear==1):
        dv_xi_plus  = compute_datavector(theta_emu, emu_xi_plus)
        dv_xi_minus = compute_datavector(theta_emu, emu_xi_minus)
    else:
        dv_xi_plus  = np.zeros(config.N_xi)
        dv_xi_minus = np.zeros(config.N_xi)
    if (config.shear_pos==1):
        dv_gammat   = compute_datavector(theta_emu, emu_gammat)
    else:
        dv_gammat   = np.zeros(config.N_ggl)
    if (config.pos_pos==1):
        dv_wtheta   = compute_datavector(theta_emu, emu_wtheta)
    else:
        dv_wtheta   = np.zeros(config.N_w)
    if (config.gk==1):
        dv_gk       = compute_datavector(theta_emu, emu_gk)
    else:
        dv_gk = np.zeros(config.N_gk)
    if (config.ks==1):
        dv_ks       = compute_datavector(theta_emu, emu_ks)
    else:
        dv_ks = np.zeros(config.N_sk)
    if (config.kk==1):
        dv_kk       = compute_datavector(theta_emu, emu_kk)
    else:
        dv_kk       = np.zeros(config.N_kk)
    
    datavector  = np.hstack([dv_xi_plus, dv_xi_minus, dv_gammat, dv_wtheta, dv_gk, dv_ks, dv_kk])

    # ============== Add shear calibration bias ======================
    if(config.probe!='wtheta'):
        m_shear_theta = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo):self.n_sample_dims-(self.n_pcas_baryon)]
        datavector  = self.add_shear_calib(m_shear_theta, datavector)
    # ============== Add baryons ======================
    if(self.n_pcas_baryon > 0.):
        baryon_Q    = theta[self.n_sample_dims-self.n_pcas_baryon:]
        datavector  = self.add_baryon_q(baryon_Q, datavector)
    # ============== Add liner galaxy bias ============
    if(config.probe!='cosmic_shear'):
        gbias_theta = theta[self.n_sample_dims-(self.n_pcas_baryon+self.source_ntomo+self.lens_ntomo): self.n_sample_dims-(self.n_pcas_baryon+self.source_ntomo)]
        datavector = self.add_bias(gbias_theta, datavector)
    return datavector
    
def ln_lkl(theta):
    model_datavector = get_data_vector_emu(theta)
    delta_dv = (model_datavector - emu_sampler.dv_obs)[emu_sampler.mask]
    return -0.5 * delta_dv @ emu_sampler.masked_inv_cov @ delta_dv        

def ln_prob(theta, temper_val=1.):
    return emu_sampler.ln_prior(theta) + temper_val * ln_lkl(theta)

#================= Run MCMC chains using emulator ==============================
print("temper_val: %2.3f"%(temper_val))

with Pool() as pool:
    sampler = emcee.EnsembleSampler(config.n_emcee_walkers, 
        emu_sampler.n_sample_dims, ln_prob, args=(temper_val,), pool=pool)
    sampler.run_mcmc(pos0, config.n_mcmc, progress=True)
    # save the sampler for debug purpose
    if (args.debug):
        _sample = sampler.get_chain(flat=True)
        _logprob= sampler.get_log_prob(flat=True)
        _sigma8 = emu_s8.predict(torch.Tensor(_sample[:,:config.n_pars_cosmo]))
        np.save(pjoin(config.traindir, f'DBG_chain_{label}_{n}.npy'), 
            np.hstack([_sample, _sigma8, _logprob[:,np.newaxis]]))

samples = sampler.get_chain(discard=config.n_burn_in, thin=config.n_thin, flat=True)

if(args.temper):
    # only save samples to explore posterior regions
    select_indices = np.random.choice(np.arange(len(samples)), replace=False, size=config.n_resample)
    next_training_samples = samples[select_indices,:-(config.n_fast_pars)]
    np.save(pjoin(config.traindir, f'samples_{label}_{n+1}.npy'), next_training_samples)
else:
    # we want the chain
    logprobs= sampler.get_log_prob(discard=config.n_burn_in, thin=config.n_thin, flat=True)
    derived_sigma8 = emu_s8.predict(torch.Tensor(samples[:,:config.n_pars_cosmo]))
    np.save(pjoin(config.chaindir, config.chainname+f'_{label}_{n}.npy'), 
        np.hstack([samples, derived_sigma8, logprobs[:,np.newaxis]]))
print("train_emulator.py: iteration %d Done!"%n)
