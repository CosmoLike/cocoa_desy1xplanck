import sys
import os
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

train_samples      = np.load(config.savedir + '/train_samples_%d.npy'%(n))
train_data_vectors = np.load(config.savedir + '/train_data_vectors_%d.npy'%(n))
    
#=================================================

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
#==============================================
torch.set_num_threads(48)
    
full_emu = NNEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std, config.mask_ones, config.nn_model)
#==============================================
N_Z_BINS = 5
N_angular_bins = 26

ggl_exclude = []

N_xi  = int((N_Z_BINS * (N_Z_BINS + 1)) // 2 * N_angular_bins)
N_ggl = int((N_Z_BINS * N_Z_BINS - len(ggl_exclude)) * N_angular_bins)
N_w   = int(N_Z_BINS * N_angular_bins)

dv_ggl = np.zeros(N_ggl)
dv_w   = np.zeros(N_w)

print("N_xi: %d"%(N_xi))

print("=======================================")
print("Training xi_plus emulator....")
emu_xi_plus = NNEmulator(config.n_dim, N_xi, config.dv_fid[:N_xi], config.dv_std[:N_xi], config.mask[:N_xi], config.nn_model)
emu_xi_plus.train(torch.Tensor(train_samples), torch.Tensor(train_data_vectors[:,:N_xi]),\
              batch_size=config.batch_size, n_epochs=config.n_epochs)
print("=======================================")
print("=======================================")
print("Training xi_minus emulator....")
emu_xi_minus = NNEmulator(config.n_dim, N_xi, config.dv_fid[N_xi:2*N_xi], config.dv_std[N_xi:2*N_xi], config.mask[N_xi:2*N_xi], config.nn_model)
emu_xi_minus.train(torch.Tensor(train_samples), torch.Tensor(train_data_vectors[:,N_xi:2*N_xi]),\
              batch_size=config.batch_size, n_epochs=config.n_epochs)
print("=======================================")

if(save_emu):
    emu_xi_plus.save(config.savedir + '/xi_p_%d'%(n))
    emu_xi_minus.save(config.savedir + '/xi_m_%d'%(n))
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
    
    dv_xi_plus  = compute_datavector(theta_emu, emu_xi_plus)
    dv_xi_minus = compute_datavector(theta_emu, emu_xi_minus)
    
    datavector  = np.hstack([dv_xi_plus, dv_xi_minus, dv_ggl, dv_w])
    # ============== Add shear calibration bias ======================
    m_shear_theta = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo):
                          self.n_sample_dims-(self.n_pcas_baryon)]
    datavector  = self.add_shear_calib(m_shear_theta, datavector)
    # ============== Add baryons ======================
    if(self.n_pcas_baryon > 0.):
        baryon_Q    = theta[self.n_sample_dims-self.n_pcas_baryon:]
        datavector  = self.add_baryon_q(baryon_Q, datavector)    
    return datavector
    
def ln_lkl(theta):
    model_datavector = get_data_vector_emu(theta)
    delta_dv = (model_datavector - emu_sampler.dv_obs)[emu_sampler.mask]
    return -0.5 * delta_dv @ emu_sampler.masked_inv_cov @ delta_dv        

def ln_prob(theta, temper_val=1.):
    return emu_sampler.ln_prior(theta) + temper_val * ln_lkl(theta)

#==============================================    
print("temper_val: %2.3f"%(temper_val))

with Pool() as pool:
    sampler = emcee.EnsembleSampler(config.n_emcee_walkers, emu_sampler.n_sample_dims, 
                                        ln_prob, args=(temper_val,), pool=pool)
    sampler.run_mcmc(pos0, config.n_mcmc, progress=True)

samples = sampler.chain[:,config.n_burn_in::config.n_thin].reshape((-1, emu_sampler.n_sample_dims))

if(temper):
    select_indices = np.random.choice(np.arange(len(samples)), replace=False, size=config.n_resample)
    next_training_samples = samples[select_indices,:-(config.n_fast_pars)]
    np.save(config.savedir + '/train_samples_%d.npy'%(n+1), next_training_samples)
else:
    np.save(config.savedir + '/' + config.chainname + '_%d.npy'%(n), samples)


