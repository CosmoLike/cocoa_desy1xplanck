import sys
import os
from os.path import join as pjoin
import numpy as np
import torch
from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler
import emcee

from multiprocessing import Pool

configfile = sys.argv[1]
n          = int(sys.argv[2])

#==============================================
temper_schedule = [0.02, 0.1, 0.2, 0.4, 0.6, 0.7, 0.9, 0.9]

try:
    temper = (int(sys.argv[3])==1)
except:
    temper = False

if(temper):
    temper_val = temper_schedule[n]
else:
    temper_val = 1.

print("temper_val: %2.3f"%(temper_val))

try:
    save_emu = (int(sys.argv[4])==1)
except:
    save_emu = False
    
#==============================================

config = Config(configfile)

#==============================================

train_samples      = np.load(pjoin(config.traindir, f'samples_{n}.npy'))
train_data_vectors = np.load(pjoin(config.traindir, f'data_vectors_{n}.npy'))
train_sigma8       = np.load(pjoin(config.traindir, f'sigma8_{n}.npy'))

#================= Clean data by chi2 cut ================================

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
total_obj    = len(select_chi_sq)
print("Total number of objects: %d"%(selected_obj))

train_data_vectors = train_data_vectors[select_chi_sq]
train_samples      = train_samples[select_chi_sq]
train_sigma8       = train_sigma8[select_chi_sq]
#================= Init emulator =============================
torch.set_num_threads(48)
# NN only hard-coded?
full_emu = NNEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std, config.mask_ones, config.nn_model)
#================= Training emulator =============================
# switch according to probes
if (config.shear_shear==1):
    print("=======================================")
    print("Training xi_plus emulator....")
    _l, _r = 0, config.N_xi
    emu_xi_plus = NNEmulator(config.n_dim, config.N_xi, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.mask[_l:_r], config.nn_model)
    emu_xi_plus.train(torch.Tensor(train_samples), 
        torch.Tensor(train_data_vectors[:,_l:_r]),
        batch_size=config.batch_size, n_epochs=config.n_epochs)
    if(save_emu):
        emu_xi_plus.save(pjoin(config.modeldir, f'xi_p_{n}_nn{config.nn_model}'))
    print("=======================================")
    print("=======================================")
    print("Training xi_minus emulator....")
    _l, _r = config.N_xi, config.N_xi*2
    emu_xi_minus = NNEmulator(config.n_dim, config.N_xi, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.mask[_l:_r], config.nn_model)
    emu_xi_minus.train(torch.Tensor(train_samples), 
        torch.Tensor(train_data_vectors[:,_l:_r]),
        batch_size=config.batch_size, n_epochs=config.n_epochs)
    if(save_emu):
        emu_xi_minus.save(pjoin(config.modeldir, f'xi_m_{n}_nn{config.nn_model}'))
    print("=======================================")
if (config.shear_pos==1):
    print("=======================================")
    print("Training gammat emulator....")
    _l, _r = config.N_xi*2, config.N_xi*2 + config.N_ggl
    emu_gammat = NNEmulator(config.n_dim, config.N_ggl, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.mask[_l:_r], config.nn_model)
    emu_gammat.train(torch.Tensor(train_samples), 
        torch.Tensor(train_data_vectors[:,_l:_r]),
        batch_size=config.batch_size, n_epochs=config.n_epochs)
    if(save_emu):
        emu_gammat.save(pjoin(config.modeldir, f'gammat_{n}_nn{config.nn_model}'))
    print("=======================================")
if (config.pos_pos==1):
    print("=======================================")
    print("Training wtheta emulator....")
    _l, _r = config.N_xi*2+config.N_ggl, config.N_xi*2+config.N_ggl+config.N_w
    emu_wtheta = NNEmulator(config.n_dim, config.N_w, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.mask[_l:_r], config.nn_model)
    emu_wtheta.train(torch.Tensor(train_samples), 
        torch.Tensor(train_data_vectors[:,_l:_r]),
        batch_size=config.batch_size, n_epochs=config.n_epochs)
    if(save_emu):
        emu_wtheta.save(pjoin(config.modeldir, f'wtheta_{n}_nn{config.nn_model}'))
    print("=======================================")
if (config.gk==1):
    print("=======================================")
    print("Training w_gk emulator....")
    _l, _r = config.N_xi*2+config.N_ggl+config.N_w, config.N_xi*2+config.N_ggl+config.N_w+config.N_gk
    emu_gk = NNEmulator(config.n_dim, config.N_gk, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.mask[_l:_r], config.nn_model)
    emu_gk.train(torch.Tensor(train_samples), 
        torch.Tensor(train_data_vectors[:,_l:_r]),
        batch_size=config.batch_size, n_epochs=config.n_epochs)
    if(save_emu):
        emu_gk.save(pjoin(config.modeldir, f'gk_{n}_nn{config.nn_model}'))
    print("=======================================")
if (config.ks==1):
    print("=======================================")
    print("Training w_sk emulator....")
    _l, _r = config.N_xi*2+config.N_ggl+config.N_w+config.N_gk, config.N_xi*2+config.N_ggl+config.N_w+config.N_gk+config.N_sk
    emu_ks = NNEmulator(config.n_dim, config.N_sk, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.mask[_l:_r], config.nn_model)
    emu_ks.train(torch.Tensor(train_samples), 
        torch.Tensor(train_data_vectors[:,_l:_r]),
        batch_size=config.batch_size, n_epochs=config.n_epochs)
    if(save_emu):
        emu_ks.save(pjoin(config.modeldir, f'ks_{n}_nn{config.nn_model}'))
    print("=======================================")
if (config.kk==1):
    print("=======================================")
    print("Training CMBL band power emulator....")
    _l, _r = config.N_xi*2+config.N_ggl+config.N_w+config.N_gk+config.N_sk, config.N_xi*2+config.N_ggl+config.N_w+config.N_gk+config.N_sk+config.N_kk
    emu_kk = NNEmulator(config.n_dim, config.N_kk, 
        config.dv_fid[_l:_r], config.dv_std[_l:_r], 
        config.mask[_l:_r], config.nn_model)
    emu_kk.train(torch.Tensor(train_samples), 
        torch.Tensor(train_data_vectors[:,_l:_r]),
        batch_size=config.batch_size, n_epochs=config.n_epochs)
    if(save_emu):
        emu_kk.save(pjoin(config.modeldir, f'kk_{n}_nn{config.nn_model}'))
    print("=======================================")
if (config.derived==1):
    print("=======================================")
    print("Training derived parameters emulator (sigma8) ....")
    emu_s8 = NNEmulator(config.n_pars_cosmo, 1, 
        config.sigma8_fid, config.sigma8_std, 
        [1], config.nn_model)
    emu_s8.train(torch.Tensor(train_samples[:,:n_pars_cosmo]), 
        torch.Tensor(train_sigma8),
        batch_size=config.batch_size, n_epochs=config.n_epochs)
    if(save_emu):
        emu_s8.save(pjoin(config.modeldir, f'sigma8_{n}_nn{config.nn_model}'))
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
    sampler = emcee.EnsembleSampler(config.n_emcee_walkers, emu_sampler.n_sample_dims, 
                                        ln_prob, args=(temper_val,), pool=pool)
    sampler.run_mcmc(pos0, config.n_mcmc, progress=True)

samples = sampler.chain[:,config.n_burn_in::config.n_thin].reshape((-1, emu_sampler.n_sample_dims))

if(temper):
    # only save samples to explore posterior regions
    select_indices = np.random.choice(np.arange(len(samples)), replace=False, size=config.n_resample)
    next_training_samples = samples[select_indices,:-(config.n_fast_pars)]
    np.save(pjoin(config.traindir, f'samples_{n+1}.npy'), next_training_samples)
else:
    # we want the chain
    np.save(pjoin(config.chaindir, config.chainname+f'_{n}.npy'), samples)
print("train_emulator.py: iteration %d Done!"%n)
