import sys, os
from tqdm import tqdm
from os.path import join as pjoin
#from mpi4py import MPI
import numpy as np
import torch
from cocoa_emu import Config
from cocoa_emu.emulator import NNEmulator
from cocoa_emu.sampling import EmuSampler
from argparse import ArgumentParser
os.environ["OMP_NUM_THREADS"] = "1"

parser = ArgumentParser()
parser.add_argument('config', type=str, help='Configuration file')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Overwrite existing model files')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Turn on debugging mode')
parser.add_argument('-thin', type=int, default=1,
                    help='Thin the validation dataset when comparing dchi2')
args = parser.parse_args()

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()

#if torch.cuda.is_available():
if False:
   device = torch.device('cuda')
   #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   device = torch.device('cpu')
   torch.set_num_interop_threads(4) # Inter-op parallelism
   torch.set_num_threads(4) # Intra-op parallelism
torch.set_default_dtype(torch.float32)

config = Config(args.config)
iss = f'{config.init_sample_type}'
label_valid = iss+f'_t{config.gtemp_v:d}_{config.gnsamp_v}'
N_sample_valid = config.gnsamp_v

### Load validation dataset
print(f'Loading validating data...')
valid_samples = np.load(pjoin(config.traindir, f'samples_{label_valid}.npy'))[::args.thin]
valid_data_vectors = np.load(pjoin(config.traindir, f'data_vectors_{label_valid}.npy'))[::args.thin]
valid_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label_valid}.npy'))[::args.thin]
mask_samples = np.all(np.isfinite(valid_sigma8), axis=1) & np.all(np.isfinite(valid_data_vectors), axis=1)
valid_samples = valid_samples[mask_samples]
valid_data_vectors = valid_data_vectors[mask_samples]
valid_sigma8 = valid_sigma8[mask_samples]
N_samples = valid_samples.shape[0]
print(f'Validation dataset loaded (thin by {args.thin} down to {N_samples})')

### TODO: see how the xi_pm emulator goes, decide whether to combine them.
### Load emulators
print(f'Loading emulator...')
probe_fmts = ["xi_pm", "gammat", "wtheta", "wgk", "wsk", "Ckk"]
emu_list = []
for i,p in enumerate(probe_fmts):
    _l, _r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
    #fn = pjoin(config.modeldir, f'{p}_nn{config.nn_model}')
    fn = pjoin(config.modeldir, f'{probe_fmts[i]}_nn{config.nn_model}_wd{config.weight_decay}_PCA{int(config.do_pca)}_dropout{config.dropout}')
    if os.path.exists(fn+".h5"):
        print(f'Reading {p} NN emulator from {fn}.h5 ...')
        emu = NNEmulator(config.n_dim, config.probe_size[i], 
            config.dv_lkl[_l:_r], config.dv_std[_l:_r],
            config.inv_cov[_l:_r,_l:_r],
            mask=config.mask_lkl[_l:_r],param_mask=config.probe_params_mask[i],
            model=config.nn_model, device=device,
            deproj_PCA=config.do_pca, lr=config.learning_rate, 
            reduce_lr=config.reduce_lr, weight_decay=config.weight_decay, 
            dropout=config.dropout, dtype="float")
        emu.load(fn)
    else:
        print(f'Can not find {p} emulator {fn}! Ignore probe {p}!')
        emu = None
    emu_list.append(emu)
emu_sampler = EmuSampler(emu_list, config)

### Load sigma_8 emulator
#fn = pjoin(config.modeldir, f'sigma8_nn{config.nn_model}')
emu_s8_fn = pjoin(config.modeldir, f'sigma8_nn{config.nn_model}_wd{config.weight_decay}_PCA{int(config.do_pca)}_dropout{config.dropout}')
if os.path.exists(emu_s8_fn+".h5"):
    print(f'Reading sigma8 NN emulator from {emu_s8_fn}.h5 ...')
    emu_s8 = NNEmulator(config.n_pars_cosmo, 1, config.sigma8_fid, 
            config.sigma8_std, np.atleast_2d(1.0/config.sigma8_std**2), 
            model=config.nn_model, device=device,
            deproj_PCA=False, lr=config.learning_rate, 
            reduce_lr=config.reduce_lr, dropout=config.dropout,
            weight_decay=config.weight_decay, dtype="float")
    emu_s8.load(emu_s8_fn)
else:
    print(f'Can not find sigma8 emulator {emu_s8_fn}!')
    emu_s8 = None

print("\n\n\n Computing dchi2...")
### Compute dchi2
dchi2_list = []
dsigma8_list = []
mv_list = []
assert valid_samples.shape[1]==config.n_dim, f'Inconsistent param dimension'+\
f'{valid_samples.shape[1]} v.s. {config.n_dim}'
for theta, dv, sigma8 in tqdm(zip(valid_samples, valid_data_vectors, valid_sigma8), total=N_samples):
    # pad fiducial values for n_fast
    theta_padded = np.hstack([theta, emu_sampler.m_shear_fid, 
        np.zeros(emu_sampler.n_pcas_baryon)])
    mv = emu_sampler.get_data_vector_emu(theta_padded, skip_fast=True)
    diff = (dv-mv)
    dchi2 = diff@config.inv_cov@diff

    # break-down dchi2s
    dchi2_breakdown = []
    for i in range(6):
        _l, _r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
        sub_dchi2 = diff[_l:_r]@config.inv_cov[_l:_r,_l:_r]@diff[_l:_r]
        dchi2_breakdown.append(sub_dchi2)
    dchi2_list.append(dchi2)
    sigma8_predict = emu_s8.predict(torch.Tensor(theta[:config.n_pars_cosmo]))[0]
    mv_list.append(mv)
    dsigma8_list.append((sigma8 - (sigma8_predict))[0])
    #print(f'dchi2 = {dchi2_list[-1]}, dsigma8 = {dsigma8_list[-1]}')
    #print("break-down dchi2s: ", dchi2_breakdown)
dchi2_list = np.array(dchi2_list)
dsigma8_list = np.array(dsigma8_list)
mv_list = np.array(mv_list)

frac_dchi2_1 = np.sum(dchi2_list>1.)/dchi2_list.shape[0]
frac_dchi2_2 = np.sum(dchi2_list>0.2)/dchi2_list.shape[0]
print(f'{frac_dchi2_1} chance of getting dchi2 > 1.0 from validation sample')
print(f'{frac_dchi2_2} chance of getting dchi2 > 0.2 from validation sample')

np.save(pjoin(config.modeldir, f'dchi2_dsigma8_validation_w0waCDM_HF_NLA_025_model{config.nn_model}_wd{config.weight_decay}_PCA{int(config.do_pca)}_dropout{config.dropout}'),
	np.vstack([dchi2_list, dsigma8_list]))
np.save(pjoin(config.modeldir, f'mv_thinned_validation_w0waCDM_HF_NLA_025_model{config.nn_model}_wd{config.weight_decay}_PCA{int(config.do_pca)}_dropout{config.dropout}'),
    np.vstack(mv_list))

print("Done!")
