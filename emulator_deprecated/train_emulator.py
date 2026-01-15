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
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Overwrite existing model files')
parser.add_argument('--double_precision', action='store_true', default=False,
                    help='Use double precision for NN model parameters')
parser.add_argument('--monitor_gradient', action='store_true', default=False,
                    help='Monitor NN model gradient during training')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_num_interop_threads(40) # Inter-op parallelism
    torch.set_num_threads(40) # Intra-op parallelism
### Precision for NN model
#torch.set_default_dtype(torch.float32)
if args.double_precision:
    #torch.set_default_dtype(torch.float64)
    _DTYPE_ = torch.float64
    _DTYPE_STRING_ = "double"
else:
    #torch.set_default_dtype(torch.float32)
    _DTYPE_ = torch.float32
    _DTYPE_STRING_ = "float"
print('Using device: ',device)
print('Torch default dtype: ', torch.get_default_dtype())

#===============================================================================
config = Config(args.config)
print(f'\n>>> Start Emulator Training\n')
if config.init_sample_type == "lhs":
    print("We don't support LHS any more!")
    exit(1)
else:
    iss = f'{config.init_sample_type}'
    label_train = iss+f'_t{config.gtemp_t}_{config.gnsamp_t}'
    label_valid = iss+f'_t{config.gtemp_v}_{config.gnsamp_v}'
    N_sample_train = config.gnsamp_t
    N_sample_valid = config.gnsamp_v
#===============================================================================
# masking data vectors with extreme values
fid_dv = np.genfromtxt('/groups/timeifler/jiachuanxu/cocoa_v4/Cocoa/projects/'+\
    'desy1xplanck/data/data_vectors/DESY3xPlanckPR4_6x2pt_Maglim_baseline_dmo_HF.simudata')[:,1]
dv_mask = [0, ]* 10 + [1, ]* 20
dv_mask = dv_mask * 60 + [0,] * 9
dv_mask = np.array(dv_mask, dtype=bool)
dv_mask = dv_mask & (fid_dv!=0.0)
#================== Loading Training & Validating Data =========================
# NOTE: training/validating data are stored in float64 precision
print(f'Loading training data!')
train_samples = np.load(pjoin(config.traindir, f'samples_{label_train}.npy')).astype(np.float64)
train_data_vectors = np.load(pjoin(config.traindir, f'data_vectors_{label_train}.npy')).astype(np.float64)
train_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label_train}.npy')).astype(np.float64)
# Assert no NaN or inf values in training data
train_mask = np.all(np.isfinite(train_sigma8), axis=1) & np.all(np.isfinite(train_data_vectors), axis=1)

print(f'Loading validating data!')
valid_samples = np.load(pjoin(config.traindir, f'samples_{label_valid}.npy')).astype(np.float64)
valid_data_vectors = np.load(pjoin(config.traindir, f'data_vectors_{label_valid}.npy')).astype(np.float64)
valid_sigma8 = np.load(pjoin(config.traindir, f'sigma8_{label_valid}.npy')).astype(np.float64)
# Assert no NaN or inf values in validating data
valid_mask = np.all(np.isfinite(valid_sigma8), axis=1) & np.all(np.isfinite(valid_data_vectors), axis=1)

### Convert to torch Tensors
### By default, the tensor will be in float32 precision although the numpy arrays are in float64
train_samples = torch.tensor(train_samples[train_mask], dtype=_DTYPE_)
train_data_vectors = torch.tensor(train_data_vectors[train_mask], dtype=_DTYPE_)
train_sigma8 = torch.tensor(train_sigma8[train_mask], dtype=_DTYPE_)
valid_samples = torch.tensor(valid_samples[valid_mask], dtype=_DTYPE_)
valid_data_vectors = torch.tensor(valid_data_vectors[valid_mask], dtype=_DTYPE_)
valid_sigma8 = torch.tensor(valid_sigma8[valid_mask], dtype=_DTYPE_)

### Testing data type of some dataset
dtype_cov = config.inv_cov.dtype
dtype_dverr = config.dv_std.dtype
dtype_dv = config.dv_lkl.dtype
dtype_train_dv = train_data_vectors.dtype
dtype_train_samp = train_samples.dtype
print(f'Data type check:')
print(f'  inv_cov: {dtype_cov}, dv_std: {dtype_dverr}, dv_lkl: {dtype_dv}')
print(f'  train_data_vectors: {dtype_train_dv}, train_samples: {dtype_train_samp}')

#================= Training emulator ===========================================
# switch according to probes
probes = ["xi_pm", "gammat", "wtheta", "wgk", "wsk", "Ckk"]
for i in range(len(config.probe_mask)):
#for i in range(1):
    print("============= Training %s Emulator ================="%(probes[i]))
    l, r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
    # NOTE: dtype only controls NN model params precision
    emu = NNEmulator(config.n_dim, config.probe_size[i], 
        config.dv_lkl[l:r], config.dv_std[l:r], 
        config.inv_cov[l:r,l:r],
        mask=config.mask_lkl[l:r], 
        param_mask=config.probe_params_mask[i], 
        model=config.nn_model, device=device,
        deproj_PCA=config.do_pca, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dropout=config.dropout, dtype=_DTYPE_STRING_)
    emu_fn = pjoin(config.modeldir, f'{probes[i]}_nn{config.nn_model}_wd{config.weight_decay}_PCA{int(config.do_pca)}_dropout{config.dropout}')
    loss_fn = pjoin(config.modeldir, f'{probes[i]}_nn{config.nn_model}_wd{config.weight_decay}_PCA{int(config.do_pca)}_dropout{config.dropout}_losses.txt')
    if (not os.path.exists(emu_fn)) or args.overwrite:
        emu.train(train_samples, train_data_vectors[:,l:r],
                valid_samples, valid_data_vectors[:,l:r],
                batch_size=config.batch_size, n_epochs=config.n_epochs, 
                loss_type=config.loss_type, save_loss_filename=loss_fn, 
                debug_grad=args.monitor_gradient)
        emu.save(emu_fn)

# skip sigma_8 training, for testing purpose
# exit(0)

# train sigma_8 emulator
if (config.derived==1):
    print("============= Training sigma8 Emulator =================")
    emu_s8 = NNEmulator(config.n_pars_cosmo, 1, 
        config.sigma8_fid, config.sigma8_std, 
        np.atleast_2d(1.0/config.sigma8_std**2), 
        model=config.nn_model, device=device,
        deproj_PCA=False, lr=config.learning_rate, 
        reduce_lr=config.reduce_lr, weight_decay=config.weight_decay,
        dropout=config.dropout, dtype=_DTYPE_STRING_)
    emu_s8_fn = pjoin(config.modeldir, f'sigma8_nn{config.nn_model}_wd{config.weight_decay}_PCA{int(config.do_pca)}_dropout{config.dropout}')
    loss_fn = pjoin(config.modeldir, f'sigma8_nn{config.nn_model}_wd{config.weight_decay}_PCA{int(config.do_pca)}_dropout{config.dropout}_losses.txt')
    if (not os.path.exists(emu_s8_fn)) or args.overwrite:
        emu_s8.train(train_samples[:,:config.n_pars_cosmo], train_sigma8,
            valid_samples[:,:config.n_pars_cosmo], valid_sigma8,
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            loss_type=config.loss_type, save_loss_filename=loss_fn,
            debug_grad=args.monitor_gradient)
        emu_s8.save(emu_s8_fn)

print(f'\n>>> Emulator Training Completed!\n')
